import os
from typing import List, Optional
import chromadb
from config import Config
from models import DocumentChunk, SearchResult
from embedder import EmbedderWithFallback

class VectorDatabaseAgent:
    """Gestiona la base de datos vectorial para almacenar y recuperar información por usuario."""

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.persist_directory, exist_ok=True)
        # Inicializamos el cliente de ChromaDB con persistencia local
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        self.embedder = EmbedderWithFallback(config.embedding_model)

        try:
            # Recuperamos la colección si ya existe
            self.collection = self.client.get_collection(config.collection_name)
            print(f"[VectorDB] Colección cargada con {self.collection.count()} chunks")
        except Exception:
            # Creamos una colección nueva si es la primera vez
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print("[VectorDB] Nueva colección creada.")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Llama al componente de embeddings para vectorizar los textos."""
        return self.embedder.encode(texts)

    def add_chunks(self, chunks: List[DocumentChunk], user_id: str):
        """Agrega nuevos fragmentos de documentos a la colección, etiquetados por usuario."""
        if not chunks:
            return

        # Evitamos duplicados comprobando los IDs existentes
        existing_ids = set(self.collection.get(ids=[chunk.chunk_id for chunk in chunks])["ids"])
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]

        if not new_chunks:
            return

        texts = [chunk.text for chunk in new_chunks]
        ids = [chunk.chunk_id for chunk in new_chunks]
        # Agregamos el user_id a los metadatos para aislamiento
        metas = []
        for chunk in new_chunks:
            metadata = chunk.metadata.copy()
            metadata["user_id"] = user_id
            metas.append(metadata)

        embeddings = self._embed(texts)

        # Agregamos los datos en lotes
        for i in range(0, len(new_chunks), 500):
            self.collection.add(
                documents=texts[i: i + 500],
                embeddings=embeddings[i: i + 500],
                metadatas=metas[i: i + 500],
                ids=ids[i: i + 500],
            )

    def search(self, query: str, user_id: str, n_results: int = 5, doc_filter: Optional[list] = None) -> SearchResult:
        """Busca fragmentos relevantes pertenecientes únicamente al usuario actual."""
        q_emb = self._embed([query])[0]

        # Siempre filtramos por user_id
        where_conditions = [{"user_id": user_id}]
        if doc_filter:
            where_conditions.append({"document_id": {"$in": doc_filter}})

        where = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]

        # Contamos cuántos documentos tiene este usuario específicamente
        user_total = self.collection.count() # Nota: count() es global, pero query filtrará

        kwargs = dict(
            query_embeddings=[q_emb],
            n_results=min(n_results, max(1, user_total)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        res = self.collection.query(**kwargs)

        if not res["documents"] or not res["documents"][0]:
            return SearchResult()

        return SearchResult(
            documents=res["documents"][0],
            metadatas=res["metadatas"][0],
            distances=res["distances"][0],
        )

    def count_user_chunks(self, user_id: str) -> int:
        """Devuelve el número de fragmentos que pertenecen a un usuario."""
        res = self.collection.get(where={"user_id": user_id}, include=[])
        return len(res["ids"])

    def list_user_documents(self, user_id: str) -> List[str]:
        """Obtiene la lista de documentos indexados de un usuario específico."""
        data = self.collection.get(where={"user_id": user_id}, include=["metadatas"])
        doc_ids = set()
        for meta in (data.get("metadatas") or []):
            if "document_id" in meta:
                doc_ids.add(meta["document_id"])
        return sorted(list(doc_ids))

    def clear_user_data(self, user_id: str):
        """Elimina solo los documentos que pertenecen al usuario especificado."""
        self.collection.delete(where={"user_id": user_id})