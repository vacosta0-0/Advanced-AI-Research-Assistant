# AI Research Assistant v3.2 🚀

Este repositorio contiene un sistema de **Asistente de Investigación Avanzado** basado en una arquitectura de **múltiples agentes inteligentes**. La aplicación permite realizar análisis multi-documento de archivos PDF utilizando técnicas de **Generación Aumentada por Recuperación (RAG)**, garantizando el aislamiento de datos por usuario.

---

## 📂 Estructura del Proyecto

A continuación se detalla la función de cada archivo dentro de la arquitectura modular del sistema:

- **`app.py`**: Punto de entrada principal. Verifica e instala dependencias y lanza la interfaz de usuario.
- **`assistant.py`**: **Orquestador Principal**. Coordina la interacción entre el agente de extracción, la base de datos vectorial y los sintetizadores de IA.
- **`config.py`**: Centraliza la configuración del sistema, incluyendo modelos de Hugging Face (Llama-3.1 por defecto), parámetros de fragmentación (`chunk_size: 600`, `overlap: 100`) y rutas de persistencia.
- **`extractor.py`**: **Agente de Extracción**. Utiliza `PyPDF2` y `pdfminer.six` para extraer texto de PDFs y dividirlo en fragmentos manejables.
- **`embedder.py`**: Genera **embeddings semánticos** con `SentenceTransformer`, con respaldo en TF-IDF.
- **`vectordb.py`**: **Agente de Almacenamiento**. Gestiona la base de datos vectorial `ChromaDB`, con aislamiento por usuario.
- **`synthesizers.py`**: **Agente de Síntesis**. Conecta con Hugging Face, OpenAI y Anthropic para generar respuestas académicas basadas en un `SYSTEM_PROMPT`.
- **`ui.py`**: Interfaz de usuario construida con **Gradio**, organizada en pestañas para consultas, gestión de archivos y configuración.
- **`models.py`**: Define estructuras de datos como `DocumentChunk` y `SearchResult`.
- **`requirements.txt`**: Lista de dependencias necesarias (Gradio, ChromaDB, Hugging Face Hub, etc.).

---

## 🏗️ Arquitectura del Sistema

El sistema se basa en flujos de trabajo especializados representados en los siguientes diagramas:

1. **Arquitectura de Componentes** – (D1)  
2. **Flujo de Indexación** – (D2)  
3. **Flujo de Consulta (RAG)** – (D3)  
4. **Diagrama de Secuencia UML** – (D4)  
5. **Ciclo de Vida del Sintetizador** – (D5)  

Cada diagrama está disponible en formato PNG con los nombres indicados (D1–D5).

---

## 📸 Evidencias de Funcionamiento (Hugging Face Spaces)

La aplicación está desplegada y operativa, con las siguientes funciones:

- **Interfaz de Consultas**: Chatbot interactivo con soporte multi-fuente. *(EV1)*  
- **Gestión de Archivos**: Panel para subir, indexar y eliminar PDFs de forma privada. *(EV2)*  
- **Configuración Dinámica**: Selector de proveedores de LLM (Hugging Face, Anthropic, OpenAI) y modelos como `Llama-3.1-8B`. *(EV3)*  

---

## 🛠️ Requisitos e Instalación

Para ejecutar este proyecto localmente:

```bash
pip install -r requirements.txt
python app.py
---

## Dependencias

| Librería | Versión | Uso |
|----------|---------|-----|
| `PyPDF2` | ≥3.0 | Extracción de texto de PDFs |
| `pdfminer.six` | ≥20221105 | Extracción robusta de PDFs complejos |
| `chromadb` | ≥0.4 | Base de datos vectorial persistente |
| `sentence-transformers` | ≥2.2 | Generación de embeddings semánticos |
| `transformers` | ≥4.30 | Modelos de lenguaje (síntesis opcional) |
| `torch` | ≥2.0 | Backend para modelos de ML |
| `numpy` | ≥1.24 | Operaciones numéricas |

---

## 🔬 Descripción Técnica

### Extracción de Texto (`PDFExtractorAgent`)
- Implementa **dos estrategias de extracción**: PyPDF2 (rápido) y PDFMiner (robusto para PDFs complejos)
- Modo `auto`: intenta PyPDF2 primero, cambia a PDFMiner si el resultado es insuficiente
- **Chunking con overlap**: divide el texto en fragmentos de tamaño configurable con solapamiento para mantener contexto entre fragmentos

### Base de Datos Vectorial (`VectorDatabaseAgent`)
- Usa **ChromaDB** con persistencia en disco
- Genera embeddings con el modelo `all-MiniLM-L6-v2` (sentence-transformers)
- Búsqueda por **similitud coseno** en espacio vectorial de alta dimensión

### Síntesis de Respuestas (`ResponseSynthesizerAgent`)
- **Modo extractivo**: selecciona y presenta fragmentos más relevantes con porcentaje de relevancia
- **Modo LLM** (opcional): usa `facebook/bart-large-cnn` para síntesis generativa si está disponible
- Degrada graciosamente al modo extractivo si el modelo LLM no puede cargarse

---

## 📊 Resultados y Evaluación

Los resultados de las pruebas y métricas de evaluación del sistema se documentarán en la carpeta `results/` conforme avance el desarrollo de la tesis.

---

## 👤 Autor

**[Victoria Acosta Sarauz]**
- Universidad: Universidad Técnica del Norte
- Carrera: Tenologías de la Información
- Director de tesis: Pablo Andrés Lnadeta López
- Año: 2025

---

## 📄 Licencia

Este proyecto es desarrollado con fines académicos como parte de un trabajo de titulación.

---

## 🙏 Reconocimientos

- [sentence-transformers](https://www.sbert.net/) por los modelos de embeddings
- [ChromaDB](https://www.trychroma.com/) por la base de datos vectorial
- [Hugging Face](https://huggingface.co/) por los modelos de lenguaje
