import os
from dataclasses import dataclass

# Configuración de caché para modelos de Hugging Face
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/st_cache")

# Modelos verificados en HF Inference Providers (marzo 2026).
# Estos modelos usan Inference Providers seleccionados automáticamente.
HF_WORKING_MODELS = {
    "Llama-3.1-8B (recomendado)":    "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.3-70B (mejor calidad)": "meta-llama/Llama-3.3-70B-Instruct",
    "gpt-oss-20b (rápido)":          "openai/gpt-oss-20b",
    "Qwen2.5-Coder-32B":             "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Mistral-Nemo-12B":              "mistralai/Mistral-Nemo-Instruct-2407",
}

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_ENV_TOKEN = os.environ.get("HF_TOKEN", "")

@dataclass
class Config:
    """Configuración central del asistente de investigación."""
    hf_token: str = _ENV_TOKEN
    hf_model: str = DEFAULT_MODEL
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    claude_model: str = "claude-sonnet-4-6"
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o-mini"
    provider: str = "huggingface"
    chunk_size: int = 600
    overlap: int = 100
    n_results: int = 5  # Aumentado para mejor contexto multi-documento
    collection_name: str = "research_papers_v32"
    persist_directory: str = "/tmp/chroma_db_v32"
    max_workers: int = 2
    embedding_model: str = "all-MiniLM-L6-v2"

CONFIG = Config()