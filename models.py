from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DocumentChunk:
    """Representa un fragmento de texto de un documento PDF."""
    text: str
    metadata: Dict
    chunk_id: str

@dataclass
class SearchResult:
    """Resultados obtenidos tras una búsqueda en la base de datos vectorial."""
    documents: List[str] = field(default_factory=list)
    metadatas: List[Dict] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)