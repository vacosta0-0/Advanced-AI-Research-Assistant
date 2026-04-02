from typing import List, Dict, Optional
from huggingface_hub import InferenceClient
from config import DEFAULT_MODEL, HF_WORKING_MODELS
from models import SearchResult

# Indicaciones iniciales (System Prompt) para guiar el comportamiento de los modelos de IA
SYSTEM_PROMPT = (
    "Eres un asistente experto en análisis de literatura académica. "
    "Tu objetivo es sintetizar información de varios artículos académicos proporcionados como contexto. "
    "Responde basándote EXCLUSIVAMENTE en los fragmentos de artículos adjuntos. "
    "Es IMPORTANTE que utilices información de múltiples fuentes si están disponibles para dar una respuesta completa. "
    "Indica claramente de qué documento (fuente) proviene cada idea importante. "
    "Si el contexto no tiene suficiente información, dilo de forma explícita. "
    "Usa un tono académico y profesional, pero accesible. "
    "Responde siempre en el mismo idioma en el que se te hace la pregunta."
)

def _build_context(result: SearchResult) -> str:
    """Prepara el contexto combinando fragmentos de múltiples fuentes para enviarlo al modelo."""
    blocks = []
    # Usamos todos los resultados disponibles (hasta el límite n_results configurado)
    for doc, metadata, distance in zip(result.documents, result.metadatas, result.distances):
        relevance = round((1 - distance) * 100, 1)
        source = metadata.get("document_id", "Desconocido")
        # Añadimos un encabezado claro para cada fragmento
        blocks.append(f"[FUENTE: {source} | Relevancia: {relevance}%]\nCONTENIDO: {doc}")

    return "\n\n---\n\n".join(blocks)

def _sources_suffix(result: SearchResult) -> str:
    """Genera una pequeña nota al pie con la lista única de documentos consultados."""
    sources = list(dict.fromkeys(metadata.get("document_id", "?") for metadata in result.metadatas))
    if not sources:
        return ""
    return f"\n\n---\n📄 Fuentes consultadas: {', '.join(sources)}"


class HuggingFaceSynthesizer:
    """Sintetizador que utiliza la API de Hugging Face de manera gratuita."""

    def __init__(self, token: str, preferred_model: Optional[str] = ""):
        self.token = token.strip() if token else ""
        self.model = (preferred_model.strip() if preferred_model else "") or DEFAULT_MODEL
        print(f"[HF] Modelo seleccionado: {self.model}")

    def synthesize(self, query: str, result: SearchResult) -> str:
        """Genera una respuesta usando modelos alojados en Hugging Face."""
        if not result.documents:
            return "⚠️ No se encontró información relevante en los documentos indexados."

        if not self.token:
            return (
                "❌ **Token de Hugging Face no configurado.**\n\n"
                "Agrega tu token como Secret `HF_TOKEN` en Settings → Variables and Secrets de tu Space, "
                "o escríbelo en la pestaña ⚙️ Configuración."
            )

        context = _build_context(result)
        user_content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Pregunta de investigación: {query}\n\n"
            f"Contexto (fragmentos de múltiples artículos académicos):\n{context}\n\n"
            f"Instrucción: Proporciona una respuesta sintética, comparando y combinando la información de las fuentes citadas si es posible."
        )

        # provider="auto" hace que HF elija el proveedor con créditos gratuitos disponible
        client = InferenceClient(token=self.token, provider="auto")
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": user_content}],
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
            return answer + _sources_suffix(result)

        except Exception as e:
            err = str(e)
            if "401" in err or "unauthorized" in err.lower():
                return "❌ **Token inválido (error 401)**. Ve a https://huggingface.co/settings/tokens."
            elif "402" in err or "credits" in err.lower():
                return "⏳ **Créditos agotados**. Intenta el próximo mes."
            elif "gated" in err.lower() or "access" in err.lower():
                return f"🔒 **Modelo restringido**: acepta sus términos en huggingface.co."
            elif "rate limit" in err.lower() or "429" in err:
                return "⏳ **Límite de peticiones alcanzado**. Espera unos minutos."
            else:
                return f"❌ **Error inesperado:**\n\n```\n{err[:500]}\n```"


class AnthropicSynthesizer:
    """Sintetizador que utiliza la API de Anthropic (Claude)."""

    def __init__(self, api_key: str, model: str):
        import anthropic as _anthropic
        self.client = _anthropic.Anthropic(api_key=api_key)
        self.model = model

    def synthesize(self, query: str, result: SearchResult) -> str:
        """Genera una respuesta basada en el modelo Claude de Anthropic."""
        if not result.documents:
            return "No se encontró información relevante."

        context = _build_context(result)
        user_msg = f"Pregunta: {query}\n\nContexto:\n{context}\n\nResponde comparando las fuentes disponibles."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text + _sources_suffix(result)


class OpenAISynthesizer:
    """Sintetizador que utiliza la API de OpenAI (GPT)."""

    def __init__(self, api_key: str, model: str):
        import openai as _openai
        self.client = _openai.OpenAI(api_key=api_key)
        self.model = model

    def synthesize(self, query: str, result: SearchResult) -> str:
        """Genera una respuesta usando un modelo de lenguaje de OpenAI."""
        if not result.documents:
            return "No se encontró información relevante."

        context = _build_context(result)
        user_msg = f"Pregunta: {query}\n\nContexto:\n{context}\n\nResponde de forma sintética a partir de las fuentes adjuntas."

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
        )
        return response.choices[0].message.content + _sources_suffix(result)