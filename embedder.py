import sys
import subprocess
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbedderWithFallback:
    """Generador de embeddings con respaldo (fallback) en TF-IDF."""

    def __init__(self, model_name: str):
        self._model = None
        self._tfidf = None
        self._tfidf_fitted = False
        self._dim = 384

        try:
            print(f"[Embedder] Cargando {model_name}...")
            # Guardamos la caché en el directorio configurado
            self._model = SentenceTransformer(model_name, cache_folder="/tmp/st_cache")
            print("[Embedder] Modelo cargado correctamente.")
        except Exception as e:
            print(f"[Embedder] Falló {model_name}: {e}")
            print("[Embedder] Usando fallback TF-IDF.")
            self._init_tfidf()

    def _init_tfidf(self):
        """Inicializa TF-IDF como sistema de respaldo para generar vectores."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            # Instalación rápida si no está disponible (estilo estudiante precavido)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "scikit-learn"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(max_features=self._dim, sublinear_tf=True)

    def _tfidf_embed(self, texts: List[str]) -> List[List[float]]:
        """Genera representaciones vectoriales sencillas mediante TF-IDF."""
        if not self._tfidf_fitted:
            self._tfidf.fit(texts)
            self._tfidf_fitted = True

        mat = self._tfidf.transform(texts).toarray()
        # Normalizamos los vectores
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        mat = mat / norms

        # Ajustamos a la dimensión esperada rellenando con ceros si es necesario
        if mat.shape[1] < self._dim:
            padding = np.zeros((mat.shape[0], self._dim - mat.shape[1]))
            mat = np.hstack([mat, padding])

        return mat[:, :self._dim].tolist()

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Codifica una lista de textos en sus correspondientes vectores."""
        if self._model is not None:
            # Usamos el modelo transformer si se cargó bien
            return self._model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

        # En caso contrario, usamos la alternativa tradicional
        return self._tfidf_embed(texts)