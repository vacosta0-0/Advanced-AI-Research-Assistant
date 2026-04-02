import re
import hashlib
import concurrent.futures
from pathlib import Path
from typing import List, Optional
import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract
from models import DocumentChunk

class PDFExtractorAgent:
    """Agente encargado de extraer texto de archivos PDF y fragmentarlo."""

    def __init__(self, chunk_size: int, overlap: int, max_workers: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_workers = max_workers

    def _extract_pypdf2(self, pdf_path: str) -> str:
        """Intenta extraer texto usando la librería PyPDF2."""
        pages = []
        with open(pdf_path, "rb") as file_handle:
            reader = PyPDF2.PdfReader(file_handle)
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text)
        return "\n\n".join(pages)

    def _extract_pdfminer(self, pdf_path: str) -> str:
        """Intenta extraer texto usando pdfminer.six."""
        return pdfminer_extract(pdf_path) or ""

    def _extract_text(self, pdf_path: str) -> str:
        """Limpia y extrae el texto de un PDF usando múltiples métodos como respaldo."""
        try:
            text = self._extract_pypdf2(pdf_path)
            # Si el texto extraído es muy corto, probamos con pdfminer
            if len(text.strip()) < 100:
                text = self._extract_pdfminer(pdf_path)
        except Exception:
            text = self._extract_pdfminer(pdf_path)

        # Limpieza básica de espacios y saltos de línea
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)
        return text.replace("\x00", "").strip()

    def _make_chunks(self, text: str, doc_id: str, source: str) -> List[DocumentChunk]:
        """Divide el texto en fragmentos (chunks) con solapamiento."""
        words = text.split()
        step = max(1, self.chunk_size - self.overlap)
        chunks = []

        for i in range(0, len(words), step):
            chunk_words = words[i: i + self.chunk_size]
            # Generamos un ID único basado en el contenido y la posición
            chunk_id = hashlib.md5(f"{doc_id}_{i}".encode()).hexdigest()

            chunks.append(DocumentChunk(
                text=" ".join(chunk_words),
                metadata={
                    "document_id": doc_id,
                    "source_file": source,
                    "chunk_index": i // step
                },
                chunk_id=chunk_id,
            ))
        return chunks

    def process_single(self, pdf_path: str, doc_id: Optional[str] = None) -> List[DocumentChunk]:
        """Procesa un solo archivo PDF."""
        doc_id = doc_id or Path(pdf_path).name
        text = self._extract_text(pdf_path)
        return self._make_chunks(text, doc_id, pdf_path)

    def process_many(self, pdf_paths: List[str], doc_ids: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Procesa múltiples PDFs en paralelo usando hilos."""
        doc_ids = doc_ids or [None] * len(pdf_paths)
        all_chunks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_single, path, d_id): path
                for path, d_id in zip(pdf_paths, doc_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    all_chunks.extend(future.result())
                except Exception as error:
                    print(f"[Extractor] Error procesando archivo: {error}")

        return all_chunks