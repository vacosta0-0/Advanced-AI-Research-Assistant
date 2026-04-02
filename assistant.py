import os
import time
from typing import List, Optional
from config import Config
from extractor import PDFExtractorAgent
from vectordb import VectorDatabaseAgent
from synthesizers import HuggingFaceSynthesizer, AnthropicSynthesizer, OpenAISynthesizer

class ResearchAssistant:
    """Orquestador principal que une extracción, base de datos y síntesis de IA por usuario."""

    def __init__(self, config: Config):
        self.config = config
        self.extractor = PDFExtractorAgent(config.chunk_size, config.overlap, config.max_workers)
        self.vector_db = VectorDatabaseAgent(config)
        self._synthesizer = None

    def _get_synthesizer(self):
        """Inicializa dinámicamente el sintetizador de IA según el proveedor elegido."""
        if self._synthesizer is not None:
            return self._synthesizer

        provider = self.config.provider
        if provider == "huggingface":
            token = self.config.hf_token or os.environ.get("HF_TOKEN", "")
            self._synthesizer = HuggingFaceSynthesizer(token, self.config.hf_model)
        elif provider == "anthropic":
            if not self.config.anthropic_api_key:
                raise ValueError("API key de Anthropic no configurada.")
            self._synthesizer = AnthropicSynthesizer(self.config.anthropic_api_key, self.config.claude_model)
        elif provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("API key de OpenAI no configurada.")
            self._synthesizer = OpenAISynthesizer(self.config.openai_api_key, self.config.openai_model)
        else:
            raise ValueError(f"Proveedor desconocido: {provider}")

        return self._synthesizer

    def add_documents(self, pdf_paths: List[str], user_id: str, doc_ids: Optional[List[str]] = None) -> str:
        """Carga e indexa nuevos archivos PDF en el sistema para el usuario actual."""
        t0 = time.time()
        # Paso 1: Extraemos el texto de los PDFs y lo fragmentamos
        chunks = self.extractor.process_many(pdf_paths, doc_ids)
        # Paso 2: Lo guardamos en la base de datos vectorial para consultas posteriores, etiquetado por user_id
        self.vector_db.add_chunks(chunks, user_id=user_id)

        elapsed = time.time() - t0
        return (f"✅ {len(pdf_paths)} doc(s) en {elapsed:.1f}s | "
                f"{len(chunks)} chunks indexados | Total usuario: {self.vector_db.count_user_chunks(user_id)}")

    def query(self, question: str, user_id: str, doc_filter: Optional[list] = None) -> str:
        """Busca fragmentos relevantes y genera una respuesta sintética mediante IA para el usuario."""
        if self.vector_db.count_user_chunks(user_id) == 0:
            return "⚠️ No tienes documentos subidos. Sube PDFs primero."

        # Paso 3: Realizamos una búsqueda semántica filtrando por usuario
        result = self.vector_db.search(question, user_id=user_id, n_results=self.config.n_results, doc_filter=doc_filter or None)
        # Paso 4: Generamos la respuesta sintética usando el sintetizador actual
        return self._get_synthesizer().synthesize(question, result)

    def list_documents(self, user_id: str) -> List[str]:
        """Consulta la lista de archivos disponibles en la base de datos de un usuario."""
        return self.vector_db.list_user_documents(user_id)

    def clear(self, user_id: str) -> str:
        """Limpia la base de datos solo para el usuario actual."""
        self.vector_db.clear_user_data(user_id)
        # Reiniciamos el sintetizador por precaución si hay algún estado dependiente de la configuración
        # Aunque aquí el user_id no afecta directamente al objeto sintetizador en sí.
        return f"🗑️ Documentos del usuario {user_id} eliminados."