"""
Advanced AI Research Assistant v3.0
Sistema Multiagente con Hugging Face (GRATIS), Anthropic y OpenAI
Procesamiento paralelo de PDFs + UI web con Gradio
"""

import os
import sys
import re
import hashlib
import concurrent.futures
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import time

# ============================================================================
# INSTALACION AUTOMATICA DE DEPENDENCIAS
# ============================================================================

def install_dependencies():
    import subprocess
    packages = [
        "huggingface_hub",
        "PyPDF2",
        "pdfminer.six",
        "chromadb",
        "sentence-transformers",
        "gradio",
        "numpy",
        "requests",
    ]
    print("Instalando dependencias...")
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("Dependencias instaladas\n")

try:
    import PyPDF2
    import chromadb
    from sentence_transformers import SentenceTransformer
    import gradio as gr
    from pdfminer.high_level import extract_text as pdfminer_extract
    from huggingface_hub import InferenceClient
except ImportError:
    install_dependencies()
    import PyPDF2
    import chromadb
    from sentence_transformers import SentenceTransformer
    import gradio as gr
    from pdfminer.high_level import extract_text as pdfminer_extract
    from huggingface_hub import InferenceClient

try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai as _openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# CONFIGURACION CENTRAL
# ============================================================================

HF_MODELS_PRIORITY = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-2b-it",
    "tiiuae/falcon-7b-instruct",
]

@dataclass
class Config:
    hf_token: str = ""
    hf_model: str = ""
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    provider: str = "huggingface"
    chunk_size: int = 800
    overlap: int = 150
    n_results: int = 5
    collection_name: str = "research_papers_v3"
    persist_directory: str = "./chroma_db_v3"
    max_workers: int = 4
    embedding_model: str = "all-MiniLM-L6-v2"

CONFIG = Config()


# ============================================================================
# MODELOS DE DATOS
# ============================================================================

@dataclass
class DocumentChunk:
    text: str
    metadata: Dict
    chunk_id: str

@dataclass
class SearchResult:
    documents: List[str] = field(default_factory=list)
    metadatas: List[Dict] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)


# ============================================================================
# AGENTE 1: EXTRACTOR PDF (PARALELO)
# ============================================================================

class PDFExtractorAgent:
    def __init__(self, chunk_size, overlap, max_workers):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_workers = max_workers

    def _extract_pypdf2(self, pdf_path):
        pages = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t and t.strip():
                    pages.append(t)
        return "\n\n".join(pages)

    def _extract_pdfminer(self, pdf_path):
        return pdfminer_extract(pdf_path) or ""

    def _extract_text(self, pdf_path):
        try:
            text = self._extract_pypdf2(pdf_path)
            if len(text.strip()) < 100:
                text = self._extract_pdfminer(pdf_path)
        except Exception:
            text = self._extract_pdfminer(pdf_path)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)
        return text.replace("\x00", "").strip()

    def _make_chunks(self, text, doc_id, source):
        words = text.split()
        step = max(1, self.chunk_size - self.overlap)
        chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i: i + self.chunk_size]
            chunk_id = hashlib.md5(f"{doc_id}_{i}".encode()).hexdigest()
            chunks.append(DocumentChunk(
                text=" ".join(chunk_words),
                metadata={"document_id": doc_id, "source_file": source, "chunk_index": i // step},
                chunk_id=chunk_id,
            ))
        return chunks

    def process_single(self, pdf_path, doc_id=None):
        doc_id = doc_id or Path(pdf_path).name
        text = self._extract_text(pdf_path)
        return self._make_chunks(text, doc_id, pdf_path)

    def process_many(self, pdf_paths, doc_ids=None):
        doc_ids = doc_ids or [None] * len(pdf_paths)
        all_chunks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self.process_single, p, d): p for p, d in zip(pdf_paths, doc_ids)}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    all_chunks.extend(fut.result())
                except Exception as e:
                    print(f"[Extractor] Error: {e}")
        return all_chunks


# ============================================================================
# AGENTE 2: BASE DE DATOS VECTORIAL
# ============================================================================

class VectorDatabaseAgent:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        print("[VectorDB] Cargando modelo de embeddings...")
        self.embedder = SentenceTransformer(config.embedding_model)
        try:
            self.collection = self.client.get_collection(config.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def _embed(self, texts):
        return self.embedder.encode(texts, batch_size=64, show_progress_bar=False).tolist()

    def add_chunks(self, chunks):
        if not chunks:
            return
        existing_ids = set(self.collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return
        texts = [c.text for c in new_chunks]
        ids = [c.chunk_id for c in new_chunks]
        metas = [c.metadata for c in new_chunks]
        embeddings = self._embed(texts)
        for i in range(0, len(new_chunks), 500):
            self.collection.add(
                documents=texts[i: i + 500],
                embeddings=embeddings[i: i + 500],
                metadatas=metas[i: i + 500],
                ids=ids[i: i + 500],
            )

    def search(self, query, n_results=5, doc_filter: list = None):
        q_emb = self._embed([query])[0]
        where = {"document_id": {"$in": doc_filter}} if doc_filter else None
        total = self.collection.count()
        kwargs = dict(
            query_embeddings=[q_emb],
            n_results=min(n_results, max(1, total)),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)
        return SearchResult(
            documents=res["documents"][0],
            metadatas=res["metadatas"][0],
            distances=res["distances"][0],
        )

    def count(self):
        return self.collection.count()

    def list_documents(self):
        data = self.collection.get()
        ids = set()
        for m in (data.get("metadatas") or []):
            if "document_id" in m:
                ids.add(m["document_id"])
        return sorted(ids)

    def clear(self):
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


# ============================================================================
# AGENTE 3: SINTETIZADOR MULTI-PROVEEDOR
# ============================================================================

SYSTEM_PROMPT = (
    "Eres un asistente experto en analisis de literatura academica. "
    "Responde preguntas de investigacion basandote EXCLUSIVAMENTE en los fragmentos "
    "de articulos academicos proporcionados como contexto. "
    "Sintetiza informacion de multiples fuentes cuando sea relevante. "
    "Indica la fuente (nombre del documento) de cada afirmacion importante. "
    "Si el contexto no contiene suficiente informacion, dilo claramente. "
    "Usa un tono academico pero comprensible. "
    "Responde siempre en el mismo idioma que la pregunta del usuario. "
    "Usa markdown para estructurar la respuesta cuando ayude a la claridad."
)

def _build_context(result):
    blocks = []
    for doc, meta, dist in zip(result.documents[:5], result.metadatas[:5], result.distances[:5]):
        relevance = round((1 - dist) * 100, 1)
        source = meta.get("document_id", "Desconocido")
        blocks.append(f"[FUENTE: {source} | Relevancia: {relevance}%]\n{doc}")
    return "\n\n---\n\n".join(blocks)

def _sources_suffix(result):
    sources = list(dict.fromkeys(m.get("document_id", "?") for m in result.metadatas[:5]))
    return f"\n\n---\n Fuentes consultadas: {', '.join(sources)}"


class HuggingFaceSynthesizer:
    """Sintetizador gratuito con deteccion automatica del mejor modelo.
    Usa chat_completion (compatible con todos los modelos modernos de HF).
    """

    def __init__(self, token, preferred_model=""):
        self.token = token
        self.model = preferred_model or self._detect_best_model(token)
        print(f"[HuggingFace] Modelo activo: {self.model}")

    @staticmethod
    def _detect_best_model(token):
        print("[HuggingFace] Detectando mejor modelo disponible...")
        client = InferenceClient(token=token)
        for model in HF_MODELS_PRIORITY:
            try:
                # Usar chat_completion que es compatible con todos los modelos
                client.chat_completion(
                    messages=[{"role": "user", "content": "Hola"}],
                    model=model,
                    max_tokens=5,
                )
                print(f"[HuggingFace] Seleccionado: {model}")
                return model
            except Exception as e:
                print(f"[HuggingFace] {model}: no disponible ({str(e)[:60]})")
        fallback = "HuggingFaceH4/zephyr-7b-beta"
        print(f"[HuggingFace] Fallback: {fallback}")
        return fallback

    def synthesize(self, query, result):
        if not result.documents:
            return "No se encontro informacion relevante en los documentos indexados."
        context = _build_context(result)
        client = InferenceClient(token=self.token)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Pregunta de investigacion: {query}\n\n"
                    f"Contexto (fragmentos de articulos academicos):\n{context}\n\n"
                    f"Proporciona una respuesta sintetica y fundamentada."
                ),
            },
        ]
        response = client.chat_completion(
            messages=messages,
            model=self.model,
            max_tokens=800,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        return answer + _sources_suffix(result)


class AnthropicSynthesizer:
    def __init__(self, api_key, model):
        if not HAS_ANTHROPIC:
            raise ImportError("pip install anthropic")
        self.client = _anthropic.Anthropic(api_key=api_key)
        self.model = model

    def synthesize(self, query, result):
        if not result.documents:
            return "No se encontro informacion relevante."
        context = _build_context(result)
        user_msg = f"Pregunta: {query}\n\nContexto:\n{context}\n\nResponde de forma sintetica."
        response = self.client.messages.create(
            model=self.model, max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text + _sources_suffix(result)


class OpenAISynthesizer:
    def __init__(self, api_key, model):
        if not HAS_OPENAI:
            raise ImportError("pip install openai")
        self.client = _openai.OpenAI(api_key=api_key)
        self.model = model

    def synthesize(self, query, result):
        if not result.documents:
            return "No se encontro informacion relevante."
        context = _build_context(result)
        user_msg = f"Pregunta: {query}\n\nContexto:\n{context}\n\nResponde de forma sintetica."
        response = self.client.chat.completions.create(
            model=self.model, max_tokens=1500,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_msg}],
        )
        return response.choices[0].message.content + _sources_suffix(result)


# ============================================================================
# ORQUESTADOR PRINCIPAL
# ============================================================================

class ResearchAssistant:
    def __init__(self, config):
        self.config = config
        self.extractor = PDFExtractorAgent(config.chunk_size, config.overlap, config.max_workers)
        self.vector_db = VectorDatabaseAgent(config)
        self._synthesizer = None

    def _get_synthesizer(self):
        if self._synthesizer is not None:
            return self._synthesizer
        p = self.config.provider
        if p == "huggingface":
            if not self.config.hf_token:
                raise ValueError("Token de Hugging Face no configurado.")
            self._synthesizer = HuggingFaceSynthesizer(self.config.hf_token, self.config.hf_model)
        elif p == "anthropic":
            if not self.config.anthropic_api_key:
                raise ValueError("API key de Anthropic no configurada.")
            self._synthesizer = AnthropicSynthesizer(self.config.anthropic_api_key, self.config.claude_model)
        elif p == "openai":
            if not self.config.openai_api_key:
                raise ValueError("API key de OpenAI no configurada.")
            self._synthesizer = OpenAISynthesizer(self.config.openai_api_key, self.config.openai_model)
        else:
            raise ValueError(f"Proveedor desconocido: {p}")
        return self._synthesizer

    def add_documents(self, pdf_paths, doc_ids=None):
        t0 = time.time()
        chunks = self.extractor.process_many(pdf_paths, doc_ids)
        self.vector_db.add_chunks(chunks)
        elapsed = time.time() - t0
        return (f"✅ {len(pdf_paths)} doc(s) en {elapsed:.1f}s | "
                f"{len(chunks)} chunks indexados | Total BD: {self.vector_db.count()}")

    def query(self, question, doc_filter: list = None):
        if self.vector_db.count() == 0:
            return "⚠️ No hay documentos. Sube PDFs primero."
        result = self.vector_db.search(question, self.config.n_results, doc_filter=doc_filter or None)
        return self._get_synthesizer().synthesize(question, result)

    def list_documents(self):
        return self.vector_db.list_documents()

    def clear(self):
        self.vector_db.clear()
        self._synthesizer = None
        return "🗑️ Base de datos limpiada."


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

def build_ui(assistant):

    def save_config(provider, hf_token, ant_key, oai_key):
        assistant.config.provider = provider
        assistant._synthesizer = None
        if provider == "huggingface":
            hf_token = hf_token.strip()
            if not hf_token:
                return "⚠️ Ingresa tu token de Hugging Face."
            assistant.config.hf_token = hf_token
            assistant.config.hf_model = ""
            return "✅ Hugging Face configurado. El modelo se detectara en la primera consulta."
        elif provider == "anthropic":
            ant_key = ant_key.strip()
            if not ant_key:
                return "⚠️ Ingresa tu API key de Anthropic."
            assistant.config.anthropic_api_key = ant_key
            return f"✅ Anthropic | Modelo: {assistant.config.claude_model}"
        elif provider == "openai":
            oai_key = oai_key.strip()
            if not oai_key:
                return "⚠️ Ingresa tu API key de OpenAI."
            assistant.config.openai_api_key = oai_key
            return f"✅ OpenAI | Modelo: {assistant.config.openai_model}"

    def toggle_panels(provider):
        return (
            gr.update(visible=(provider == "huggingface")),
            gr.update(visible=(provider == "anthropic")),
            gr.update(visible=(provider == "openai")),
        )

    def upload_pdfs(files, progress=gr.Progress()):
        if not files:
            return "⚠️ No se seleccionaron archivos."
        paths = [f.name for f in files]
        progress(0, desc="Procesando PDFs en paralelo...")
        msg = assistant.add_documents(paths)
        progress(1)
        return msg

    def ask(question, history, selected_docs=None):
        if not question.strip():
            return history, ""
        try:
            doc_filter = selected_docs if selected_docs else None
            answer = assistant.query(question, doc_filter=doc_filter)
        except ValueError as e:
            answer = f"❌ {e}"
        except Exception as e:
            answer = f"❌ Error: {e}"
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    def refresh_docs():
        docs = assistant.list_documents()
        return "\n".join(f"- {d}" for d in docs) if docs else "*(No hay documentos aun)*"

    def clear_db():
        return assistant.clear(), "*(Base de datos vacia)*"

    with gr.Blocks(title="AI Research Assistant") as demo:

        gr.HTML("""
        <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;
                    border-radius:12px;padding:28px 36px;margin-bottom:20px">
            <h1 style="color:#f1f5f9;font-size:1.9rem;font-weight:700;margin:0 0 6px 0">
                🔬 AI Research Assistant
            </h1>
            <p style="color:#94a3b8;margin:0;font-size:0.95rem">
                Analisis semantico de literatura academica &middot; v3.0 &middot;
                <span style="color:#34d399;font-weight:600">Hugging Face GRATIS</span>
            </p>
        </div>
        """)

        with gr.Tabs():

            # TAB 1 - Configuracion
            with gr.Tab("⚙️ Configuracion"):
                with gr.Row():
                    with gr.Column(scale=2):
                        provider_radio = gr.Radio(
                            choices=["huggingface", "anthropic", "openai"],
                            value="huggingface",
                            label="Proveedor de IA",
                        )

                        with gr.Group(visible=True) as panel_hf:
                            gr.HTML("""
                            <div style="background:#064e3b;border:1px solid #059669;border-radius:8px;
                                        padding:10px 14px;margin:8px 0">
                                <b style="color:#34d399">GRATIS</b>
                                <span style="color:#6ee7b7;font-size:0.9rem">
                                 - Hasta ~1000 consultas/dia con cuenta gratuita
                                </span>
                            </div>
                            """)
                            hf_token_input = gr.Textbox(
                                label="Token de Hugging Face",
                                placeholder="hf_...",
                                type="password",
                                info="huggingface.co/settings/tokens -> New token -> tipo Read",
                            )

                        with gr.Group(visible=False) as panel_ant:
                            ant_key_input = gr.Textbox(
                                label="Anthropic API Key",
                                placeholder="sk-ant-...",
                                type="password",
                                info="console.anthropic.com",
                            )
                            claude_radio = gr.Radio(
                                choices=["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"],
                                value="claude-sonnet-4-6",
                                label="Modelo Claude",
                            )
                            def upd_claude(m):
                                assistant.config.claude_model = m
                                assistant._synthesizer = None
                            claude_radio.change(upd_claude, inputs=claude_radio)

                        with gr.Group(visible=False) as panel_oai:
                            oai_key_input = gr.Textbox(
                                label="OpenAI API Key",
                                placeholder="sk-...",
                                type="password",
                                info="platform.openai.com/api-keys",
                            )
                            oai_model_radio = gr.Radio(
                                choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                                value="gpt-4o-mini",
                                label="Modelo OpenAI",
                            )
                            def upd_oai(m):
                                assistant.config.openai_model = m
                                assistant._synthesizer = None
                            oai_model_radio.change(upd_oai, inputs=oai_model_radio)

                        save_btn = gr.Button("💾 Guardar y conectar", variant="primary")
                        config_status = gr.Textbox(label="Estado", interactive=False)

                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background:#1e293b;border-radius:10px;padding:20px;border:1px solid #334155">
                            <h3 style="color:#f1f5f9;margin-top:0">Guia rapida (GRATIS)</h3>
                            <ol style="color:#94a3b8;font-size:0.9rem;padding-left:18px;line-height:2">
                                <li>Entra a <b style="color:#e2e8f0">huggingface.co</b></li>
                                <li>Crea cuenta gratuita</li>
                                <li><b style="color:#e2e8f0">Settings &rarr; Access Tokens</b></li>
                                <li>Nuevo token tipo <b style="color:#e2e8f0">Read</b></li>
                                <li>Pegalo en el campo de arriba</li>
                                <li>Guarda y sube tus PDFs</li>
                            </ol>
                            <div style="background:#0f172a;border-radius:6px;padding:10px;margin-top:12px">
                                <p style="color:#64748b;font-size:0.82rem;margin:0">
                                    El sistema detecta automaticamente el mejor modelo
                                    disponible: Mistral-7B, Llama 3.1, Zephyr, Phi-3, etc.
                                </p>
                            </div>
                        </div>
                        """)

                provider_radio.change(toggle_panels, inputs=provider_radio,
                                      outputs=[panel_hf, panel_ant, panel_oai])
                save_btn.click(
                    save_config,
                    inputs=[provider_radio, hf_token_input, ant_key_input, oai_key_input],
                    outputs=config_status,
                )

            # TAB 2 - Documentos
            with gr.Tab("📄 Documentos"):
                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            label="Arrastra tus PDFs aqui",
                            file_types=[".pdf"],
                            file_count="multiple",
                        )
                        upload_btn = gr.Button("⬆️ Indexar documentos", variant="primary")
                        upload_status = gr.Textbox(label="Estado", interactive=False)
                    with gr.Column(scale=1):
                        doc_list = gr.Markdown("*(No hay documentos aun)*")
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Actualizar")
                            clear_btn = gr.Button("🗑️ Limpiar BD", variant="stop")
                        clear_status = gr.Textbox(label="", interactive=False)

                upload_btn.click(upload_pdfs, inputs=file_upload, outputs=upload_status).then(
                    refresh_docs, outputs=doc_list).then(
                    refresh_checklist, outputs=doc_checklist)
                refresh_btn.click(refresh_docs, outputs=doc_list)
                clear_btn.click(clear_db, outputs=[clear_status, doc_list])

            # TAB 3 - Consultas
            with gr.Tab("💬 Consultas"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220):
                        gr.HTML("""
                        <div style="color:#94a3b8;font-size:0.85rem;font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.05em;
                                    padding:8px 0 6px;border-bottom:1px solid #334155;margin-bottom:8px">
                            📂 Fuentes activas
                        </div>
                        """)
                        doc_checklist = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="",
                            info="Selecciona los documentos a consultar. Si no seleccionas ninguno, se usan todos.",
                        )
                        select_all_btn = gr.Button("Seleccionar todas", size="sm", variant="secondary")
                        deselect_all_btn = gr.Button("Deseleccionar todas", size="sm")

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=420, show_label=False, avatar_images=("👤", "🤖"))
                        with gr.Row():
                            question_input = gr.Textbox(
                                placeholder="Ej: ¿Cuales son las metodologias utilizadas?",
                                label="", scale=5, container=False,
                            )
                            send_btn = gr.Button("Enviar", scale=1, variant="primary")

                        gr.Examples(
                            examples=[
                                ["¿Cuales son las conclusiones principales?"],
                                ["¿Que metodologias experimentales se utilizaron?"],
                                ["¿Que limitaciones mencionan los autores?"],
                                ["Resume los hallazgos mas relevantes."],
                                ["¿Que trabajos futuros proponen?"],
                            ],
                            inputs=question_input,
                        )

                def refresh_checklist():
                    docs = assistant.list_documents()
                    return gr.update(choices=docs, value=docs)

                def select_all_docs():
                    docs = assistant.list_documents()
                    return gr.update(choices=docs, value=docs)

                def deselect_all_docs():
                    docs = assistant.list_documents()
                    return gr.update(choices=docs, value=[])

                select_all_btn.click(select_all_docs, outputs=doc_checklist)
                deselect_all_btn.click(deselect_all_docs, outputs=doc_checklist)
                send_btn.click(ask, inputs=[question_input, chatbot, doc_checklist], outputs=[chatbot, question_input])
                question_input.submit(ask, inputs=[question_input, chatbot, doc_checklist], outputs=[chatbot, question_input])

        gr.HTML("""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                    padding:10px 16px;margin-top:8px;font-size:0.82rem;color:#64748b">
            AI Research Assistant v3.0 | ChromaDB local | Embeddings: all-MiniLM-L6-v2 |
            HuggingFace Inference API (gratis)
        </div>
        """)

    return demo


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Research Assistant v3.0 - Hugging Face Edition")
    print("=" * 60)
    assistant = ResearchAssistant(CONFIG)
    ui = build_ui(assistant)
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
