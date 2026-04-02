"""
Advanced AI Research Assistant - Sistema Multiagente
Sistema de análisis automatizado de artículos académicos con arquitectura modular
VERSIÓN LISTA PARA USAR - Con instalación automática de dependencias
"""

import os
import sys
import subprocess
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib

# ============================================================================
# INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS
# ============================================================================

def install_dependencies():
    """Instala automáticamente todas las dependencias necesarias"""
    print("="*60)
    print("INSTALANDO DEPENDENCIAS DEL SISTEMA")
    print("="*60)
    
    packages = [
        'PyPDF2',
        'pdfminer.six',
        'chromadb',
        'sentence-transformers',
        'transformers',
        'torch',
        'numpy'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('.', '_').replace('-', '_'))
            print(f"✓ {package} ya está instalado")
        except ImportError:
            print(f"⚙ Instalando {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
                print(f"✓ {package} instalado exitosamente")
            except Exception as e:
                print(f"✗ Error instalando {package}: {e}")
    
    print("\n")

# Ejecutar instalación al importar
try:
    import PyPDF2
    import chromadb
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
except ImportError:
    install_dependencies()
    # Reintentar importación
    import PyPDF2
    import chromadb
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

from pdfminer.high_level import extract_text as pdfminer_extract
from chromadb.config import Settings


# ============================================================================
# AGENTE 1: EXTRACTOR DE TEXTO DE PDFs
# ============================================================================

@dataclass
class DocumentChunk:
    """Estructura para almacenar fragmentos de documento"""
    text: str
    metadata: Dict
    chunk_id: str
    page_number: Optional[int] = None


class PDFExtractorAgent:
    """
    Agente especializado en extracción de texto desde PDFs.
    Implementa múltiples estrategias de extracción para mayor robustez.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extraction_methods = ['pypdf2', 'pdfminer']
    
    def extract_text_pypdf2(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extracción usando PyPDF2"""
        text_parts = []
        metadata = {"method": "pypdf2", "pages": 0}
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata["pages"] = len(reader.pages)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
        
        return "\n\n".join(text_parts), metadata
    
    def extract_text_pdfminer(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extracción usando PDFMiner (más robusto para PDFs complejos)"""
        text = pdfminer_extract(pdf_path)
        metadata = {
            "method": "pdfminer",
            "length": len(text)
        }
        return text, metadata
    
    def extract_from_pdf(self, pdf_path: str, method: str = 'auto') -> str:
        """
        Extrae texto del PDF usando el método especificado.
        Si method='auto', intenta múltiples métodos.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")
        
        if method == 'auto':
            # Intentar con PyPDF2 primero, luego PDFMiner
            try:
                text, _ = self.extract_text_pypdf2(pdf_path)
                if len(text.strip()) < 100:  # Si el texto es muy corto, intentar otro método
                    text, _ = self.extract_text_pdfminer(pdf_path)
            except Exception as e:
                print(f"PyPDF2 falló, usando PDFMiner: {e}")
                text, _ = self.extract_text_pdfminer(pdf_path)
        elif method == 'pypdf2':
            text, _ = self.extract_text_pypdf2(pdf_path)
        else:
            text, _ = self.extract_text_pdfminer(pdf_path)
        
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído"""
        # Eliminar saltos de línea innecesarios
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Eliminar espacios múltiples
        text = re.sub(r' +', ' ', text)
        # Eliminar caracteres especiales problemáticos
        text = text.replace('\x00', '')
        return text.strip()
    
    def create_chunks(self, text: str, document_id: str, metadata: Dict = None) -> List[DocumentChunk]:
        """
        Divide el texto en fragmentos (chunks) con overlap para mantener contexto.
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Crear ID único para el chunk
            chunk_id = hashlib.md5(
                f"{document_id}_{i}".encode()
            ).hexdigest()
            
            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": i // (self.chunk_size - self.overlap),
                "total_words": len(chunk_words)
            }
            
            if metadata:
                chunk_metadata.update(metadata)
            
            chunks.append(DocumentChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def process_document(self, pdf_path: str, document_id: Optional[str] = None) -> List[DocumentChunk]:
        """
        Pipeline completo: extrae texto y lo divide en chunks.
        """
        if document_id is None:
            document_id = os.path.basename(pdf_path)
        
        print(f"[Extractor] Procesando: {document_id}")
        text = self.extract_from_pdf(pdf_path)
        print(f"[Extractor] Texto extraído: {len(text)} caracteres")
        
        chunks = self.create_chunks(
            text, 
            document_id, 
            metadata={"source_file": pdf_path}
        )
        print(f"[Extractor] Generados {len(chunks)} chunks")
        
        return chunks


# ============================================================================
# AGENTE 2: GESTOR DE BASE DE DATOS VECTORIAL
# ============================================================================

class VectorDatabaseAgent:
    """
    Agente especializado en almacenamiento y recuperación semántica.
    Utiliza ChromaDB y embeddings para búsquedas vectoriales.
    """
    
    def __init__(self, collection_name: str = "research_papers", 
                 persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Crear directorio si no existe
        os.makedirs(persist_directory, exist_ok=True)
        
        # Inicializar ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Cargar modelo de embeddings
        print("[VectorDB] Cargando modelo de embeddings...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Crear o cargar colección
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"[VectorDB] Colección '{collection_name}' cargada ({self.collection.count()} chunks)")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Academic research papers"}
            )
            print(f"[VectorDB] Colección '{collection_name}' creada")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings vectoriales para los textos"""
        return self.embedding_model.encode(texts).tolist()
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Añade chunks de documentos a la base vectorial.
        """
        if not chunks:
            return
        
        print(f"[VectorDB] Añadiendo {len(chunks)} chunks a la base de datos...")
        
        # Preparar datos
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generar embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Añadir a ChromaDB
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[VectorDB] Documentos añadidos exitosamente")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """
        Realiza búsqueda semántica en la base vectorial.
        """
        print(f"[VectorDB] Buscando: '{query[:50]}...'")
        
        # Generar embedding de la consulta
        query_embedding = self.generate_embeddings([query])[0]
        
        # Buscar en ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"[VectorDB] Encontrados {len(results['documents'][0])} resultados")
        
        return {
            "documents": results['documents'][0],
            "metadatas": results['metadatas'][0],
            "distances": results['distances'][0]
        }
    
    def get_collection_stats(self) -> Dict:
        """Obtiene estadísticas de la colección"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory
        }
    
    def clear_collection(self) -> None:
        """Limpia todos los documentos de la colección"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Academic research papers"}
        )
        print(f"[VectorDB] Colección limpiada")


# ============================================================================
# AGENTE 3: SINTETIZADOR DE RESPUESTAS
# ============================================================================

class ResponseSynthesizerAgent:
    """
    Agente especializado en síntesis de respuestas.
    Contextualiza la información recuperada y genera respuestas coherentes.
    """
    
    def __init__(self):
        print("[Synthesizer] Inicializado (modo extractivo)")
        self.use_llm = False
        
        # Intentar cargar modelo LLM (opcional)
        try:
            print("[Synthesizer] Intentando cargar modelo de lenguaje...")
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=-1
            )
            self.use_llm = True
            print("[Synthesizer] Modelo LLM cargado exitosamente")
        except Exception as e:
            print(f"[Synthesizer] No se pudo cargar LLM: {e}")
            print("[Synthesizer] Usando modo extractivo simple")
    
    def synthesize_response(self, query: str, search_results: Dict, 
                           max_length: int = 500) -> str:
        """
        Sintetiza una respuesta coherente basada en los resultados de búsqueda.
        """
        documents = search_results['documents']
        metadatas = search_results['metadatas']
        distances = search_results['distances']
        
        if not documents:
            return "❌ No se encontró información relevante para responder la consulta."
        
        # Si tenemos modelo LLM, usarlo
        if self.use_llm:
            return self._synthesize_with_llm(query, documents, metadatas, distances, max_length)
        else:
            return self._synthesize_extractive(query, documents, metadatas, distances)
    
    def _synthesize_with_llm(self, query: str, documents: List[str], 
                            metadatas: List[Dict], distances: List[float],
                            max_length: int) -> str:
        """Síntesis usando modelo de lenguaje"""
        context_parts = []
        for i, (doc, meta, dist) in enumerate(zip(documents[:3], metadatas[:3], distances[:3])):
            relevance = 1 - dist
            source = meta.get('document_id', 'Unknown')
            context_parts.append(f"[Fuente {i+1}: {source}, Relevancia: {relevance:.2f}]\n{doc[:500]}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Basándose en el siguiente contexto de artículos académicos, responde la pregunta.

Pregunta: {query}

Contexto:
{context}

Respuesta:"""
        
        try:
            response = self.summarizer(
                prompt,
                max_length=max_length,
                min_length=50,
                do_sample=False
            )
            
            answer = response[0]['summary_text']
            sources = set([meta.get('document_id', 'Unknown') for meta in metadatas[:3]])
            answer += f"\n\n📚 Fuentes: {', '.join(sources)}"
            
            return answer
        except Exception as e:
            print(f"[Synthesizer] Error en LLM: {e}")
            return self._synthesize_extractive(query, documents, metadatas, distances)
    
    def _synthesize_extractive(self, query: str, documents: List[str],
                               metadatas: List[Dict], distances: List[float]) -> str:
        """Respuesta extractiva sin modelo de lenguaje"""
        response = f"📊 **Respuesta a:** {query}\n\n"
        
        for i, (doc, meta, dist) in enumerate(zip(documents[:3], metadatas[:3], distances[:3]), 1):
            source = meta.get('document_id', 'Unknown')
            relevance = (1 - dist) * 100
            excerpt = doc[:400] + "..." if len(doc) > 400 else doc
            
            response += f"**[{i}] De '{source}' (Relevancia: {relevance:.1f}%)**\n"
            response += f"{excerpt}\n\n"
        
        return response


# ============================================================================
# SISTEMA INTEGRADO - ORQUESTADOR DE AGENTES
# ============================================================================

class AdvancedAIResearchAssistant:
    """
    Sistema principal que orquesta los tres agentes especializados.
    Proporciona interfaz unificada para el usuario.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200,
                 collection_name: str = "research_papers",
                 persist_directory: str = "./chroma_db"):
        
        print("\n" + "="*60)
        print("ADVANCED AI RESEARCH ASSISTANT")
        print("Sistema Multiagente para Análisis de Literatura Académica")
        print("="*60 + "\n")
        
        # Inicializar agentes
        self.pdf_extractor = PDFExtractorAgent(chunk_size, overlap)
        self.vector_db = VectorDatabaseAgent(collection_name, persist_directory)
        self.synthesizer = ResponseSynthesizerAgent()
        
        print("\n✓ Sistema inicializado correctamente\n")
    
    def add_document(self, pdf_path: str, document_id: Optional[str] = None) -> None:
        """
        Pipeline completo: extrae, procesa y almacena un documento PDF.
        """
        print(f"\n{'='*60}")
        print(f"AÑADIENDO DOCUMENTO: {os.path.basename(pdf_path)}")
        print(f"{'='*60}\n")
        
        try:
            # Agente 1: Extraer y dividir en chunks
            chunks = self.pdf_extractor.process_document(pdf_path, document_id)
            
            # Agente 2: Almacenar en base vectorial
            self.vector_db.add_documents(chunks)
            
            print(f"\n✓ Documento procesado e indexado exitosamente\n")
        except Exception as e:
            print(f"\n✗ Error procesando documento: {e}\n")
    
    def query(self, question: str, n_results: int = 5) -> str:
        """
        Pipeline de consulta: busca y sintetiza respuesta.
        """
        print(f"\n{'='*60}")
        print(f"CONSULTA: {question}")
        print(f"{'='*60}\n")
        
        try:
            # Agente 2: Búsqueda semántica
            search_results = self.vector_db.search(question, n_results)
            
            # Agente 3: Sintetizar respuesta
            response = self.synthesizer.synthesize_response(question, search_results)
            
            print(f"\n{'='*60}")
            print("RESPUESTA GENERADA")
            print(f"{'='*60}\n")
            print(response)
            print()
            
            return response
        except Exception as e:
            error_msg = f"✗ Error procesando consulta: {e}"
            print(f"\n{error_msg}\n")
            return error_msg
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del sistema"""
        return self.vector_db.get_collection_stats()
    
    def list_documents(self) -> List[str]:
        """Lista todos los documentos en la base de datos"""
        try:
            all_docs = self.vector_db.collection.get()
            doc_ids = set()
            if all_docs and 'metadatas' in all_docs:
                for meta in all_docs['metadatas']:
                    if 'document_id' in meta:
                        doc_ids.add(meta['document_id'])
            return sorted(list(doc_ids))
        except:
            return []
    
    def clear_database(self) -> None:
        """Limpia la base de datos"""
        self.vector_db.clear_collection()
        print("✓ Base de datos limpiada")


# ============================================================================
# INTERFAZ DE USUARIO INTERACTIVA
# ============================================================================

def interactive_menu():
    """Menú interactivo para usar el sistema fácilmente"""
    
    print("\n" + "="*60)
    print("BIENVENIDO AL ADVANCED AI RESEARCH ASSISTANT")
    print("="*60)
    
    # Inicializar sistema
    assistant = AdvancedAIResearchAssistant()
    
    while True:
        print("\n" + "-"*60)
        print("MENÚ PRINCIPAL")
        print("-"*60)
        print("1. Añadir documento PDF")
        print("2. Realizar consulta")
        print("3. Ver estadísticas")
        print("4. Listar documentos")
        print("5. Limpiar base de datos")
        print("6. Salir")
        print("-"*60)
        
        choice = input("\nSeleccione una opción (1-6): ").strip()
        
        if choice == '1':
            pdf_path = input("\nRuta del archivo PDF: ").strip()
            doc_id = input("ID del documento (Enter para usar nombre del archivo): ").strip()
            doc_id = doc_id if doc_id else None
            assistant.add_document(pdf_path, doc_id)
            
        elif choice == '2':
            if assistant.get_stats()['total_chunks'] == 0:
                print("\n⚠ No hay documentos en la base de datos. Añada documentos primero.")
                continue
            question = input("\nIngrese su pregunta: ").strip()
            if question:
                assistant.query(question)
            
        elif choice == '3':
            stats = assistant.get_stats()
            print("\n📊 ESTADÍSTICAS DEL SISTEMA")
            print("-"*60)
            print(f"Colección: {stats['collection_name']}")
            print(f"Chunks almacenados: {stats['total_chunks']}")
            print(f"Directorio: {stats['persist_directory']}")
            
        elif choice == '4':
            docs = assistant.list_documents()
            print("\n📚 DOCUMENTOS EN LA BASE DE DATOS")
            print("-"*60)
            if docs:
                for i, doc in enumerate(docs, 1):
                    print(f"{i}. {doc}")
            else:
                print("No hay documentos en la base de datos")
        
        elif choice == '5':
            confirm = input("\n⚠ ¿Está seguro de limpiar la base de datos? (si/no): ").strip().lower()
            if confirm == 'si':
                assistant.clear_database()
        
        elif choice == '6':
            print("\n¡Hasta luego! 👋\n")
            break
        
        else:
            print("\n❌ Opción inválida. Intente nuevamente.")


# ============================================================================
# EJEMPLO PROGRAMÁTICO
# ============================================================================

def example_usage():
    """
    Ejemplo de uso programático del sistema
    """
    
    # Inicializar sistema
    assistant = AdvancedAIResearchAssistant(
        chunk_size=800,
        overlap=150,
        collection_name="my_research_papers"
    )
    
    # Añadir documentos
    print("\n📝 EJEMPLO: Añadiendo documentos...")
    print("Para usar el sistema, proporcione rutas a PDFs reales:")
    print("  assistant.add_document('paper1.pdf', document_id='Smith2023_ML')")
    print("  assistant.add_document('paper2.pdf', document_id='Jones2024_NLP')")
    
    # Realizar consultas
    print("\n🔍 EJEMPLO: Realizando consultas...")
    print("Ejemplos de consultas:")
    questions = [
        "¿Cuáles son las principales técnicas de procesamiento de lenguaje natural?",
        "¿Qué metodologías se utilizaron en los experimentos?",
        "¿Cuáles son las conclusiones principales de los estudios?"
    ]
    for q in questions:
        print(f"  - {q}")
    
    # Estadísticas
    stats = assistant.get_stats()
    print("\n📊 Estadísticas actuales:")
    print(f"  - Chunks almacenados: {stats['total_chunks']}")
    
    return assistant


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SISTEMA LISTO PARA USAR")
    print("="*60)
    print("\nOpciones de uso:")
    print("1. Modo interactivo: interactive_menu()")
    print("2. Modo programático: example_usage()")
    print("\n")
    
    # Descomentar la opción deseada:
    
    # OPCIÓN A: Modo interactivo (recomendado para usuarios)
    interactive_menu()
    
    # OPCIÓN B: Modo programático (para desarrolladores)
    # assistant = example_usage()
