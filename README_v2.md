# 🔬 AI Research Assistant — v3.0 (Rama: v2-multiproveedor)

> **Evolución del sistema base** → Arquitectura avanzada con UI web, procesamiento paralelo y soporte multi-proveedor de IA.

---

## 🆚 ¿Qué cambió respecto a la versión base (`main`)?

| Característica | `main` (v1 básica) | `v2-multiproveedor` (v3.0) |
|---|---|---|
| **Interfaz** | Menú de texto en terminal | UI web con Gradio |
| **Proveedor IA** | Solo BART local | Hugging Face gratis + Anthropic + OpenAI |
| **Procesamiento PDFs** | Secuencial (uno a uno) | **Paralelo** con `ThreadPoolExecutor` |
| **Embeddings** | Por lotes básicos | `batch_size=64`, sin barra de progreso |
| **Deduplicación** | No | Sí (evita reindexar chunks existentes) |
| **Filtro por documento** | No | Sí (consultas sobre PDFs específicos) |
| **Distancia vectorial** | Default ChromaDB | Coseno explícito (`hnsw:space: cosine`) |
| **Modelos soportados** | BART (local) | Mistral-7B, Llama 3.1, Zephyr, Claude, GPT |
| **Detección automática** | No | Sí (detecta el mejor modelo HF disponible) |
| **Configuración** | Hardcoded | Dataclass `Config` centralizada |

---

## 🏗️ Arquitectura v3.0

```
┌──────────────────────────────────────────────────────────────────┐
│                    Gradio UI (build_ui)                          │
│          ⚙️ Config │ 📄 Documentos │ 💬 Consultas               │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                  ResearchAssistant (Orquestador)                  │
├──────────────────┬───────────────────────┬───────────────────────┤
│ PDFExtractorAgent│  VectorDatabaseAgent  │  Synthesizer (multi)  │
│                  │                       │                        │
│ - PyPDF2         │ - ChromaDB coseno     │ HuggingFaceSynthesizer│
│ - PDFMiner       │ - Deduplicación       │ AnthropicSynthesizer  │
│ - ThreadPool     │ - Filtro por doc      │ OpenAISynthesizer     │
│ - Chunking       │ - Batch embeddings    │ (detección automática)│
└──────────────────┴───────────────────────┴───────────────────────┘
```

---

## 🗂️ Estructura de esta rama

```
advanced-ai-research-assistant/          ← mismo repositorio
│
├── src/
│   └── research_assistant.py           ← código v3.0 (esta rama)
│
├── data/
│   ├── raw/                            # PDFs originales
│   ├── processed/                      # Datos procesados
│   └── external/                       # Datos externos
│
├── chroma_db_v3/                       # BD vectorial v3 (auto-generada)
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── requirements.txt                    # Dependencias (incluye Gradio + HF)
├── .gitignore
└── README.md                           ← este archivo
```

---

## ⚙️ Instalación

```bash
# Clonar y cambiar a esta rama
git clone https://github.com/TU_USUARIO/advanced-ai-research-assistant.git
cd advanced-ai-research-assistant
git checkout v2-multiproveedor

# Entorno virtual
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# Dependencias
pip install -r requirements.txt
```

---

## 🚀 Uso

```bash
python src/research_assistant.py
```

Abre automáticamente la UI en `http://localhost:7860`

### Pasos en la interfaz:

1. **⚙️ Configuración** → Selecciona proveedor e ingresa tu API key
   - **Hugging Face** (gratis): crea cuenta en huggingface.co → Settings → Access Tokens
   - **Anthropic**: console.anthropic.com
   - **OpenAI**: platform.openai.com/api-keys

2. **📄 Documentos** → Arrastra tus PDFs → clic en "Indexar"

3. **💬 Consultas** → Selecciona documentos a consultar y escribe tu pregunta

---

## 📦 Dependencias adicionales vs. v1

```
gradio>=3.50.0          # UI web
huggingface_hub>=0.20   # Inference API gratuita
requests>=2.31.0        # HTTP para APIs

# Opcionales (según proveedor elegido)
anthropic>=0.20.0       # Claude
openai>=1.10.0          # GPT
```

---

## 🆓 Uso con Hugging Face (sin costo)

El sistema detecta automáticamente el mejor modelo disponible en este orden:

1. `mistralai/Mistral-7B-Instruct-v0.3`
2. `meta-llama/Meta-Llama-3.1-8B-Instruct`
3. `HuggingFaceH4/zephyr-7b-beta`
4. `microsoft/Phi-3-mini-4k-instruct`
5. `google/gemma-2-2b-it`
6. `tiiuae/falcon-7b-instruct`

La cuenta gratuita permite ~1000 consultas/día.

---

## 👤 Autor

**[Tu Nombre Completo]**
- Universidad: [Nombre de tu universidad]
- Carrera: [Tu carrera]
- Director de tesis: [Nombre del director]
- Año: 2025

---

## 🔗 Ver también

- Rama `main` → versión base del sistema (terminal, sin UI)
- Comparar: `git diff main v2-multiproveedor`
