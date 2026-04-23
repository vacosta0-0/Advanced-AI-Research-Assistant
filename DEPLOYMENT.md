# Guía de Despliegue — AI Research Assistant v3.2

Este documento describe cómo replicar el sistema completo en Hugging Face Spaces desde cero.

---

## Requisitos previos

- Cuenta gratuita en [Hugging Face](https://huggingface.co)
- Cuenta gratuita en [Groq](https://console.groq.com) para obtener la API key
- Cuenta opcional en [Hugging Face](https://huggingface.co/settings/tokens) para el token HF

---

## Paso 1 — Crear el Space

1. Inicia sesión en huggingface.co
2. Ve a **Spaces → Create new Space**
3. Configura:
   - **Space name:** `advanced-ai-research-assistant`
   - **SDK:** `Gradio`
   - **Visibility:** `Public`
4. Haz clic en **Create Space**

---

## Paso 2 — Subir los archivos del proyecto

Sube los siguientes archivos desde la rama `v3-huggingface`:

```
app.py
assistant.py
config.py
crew_agents.py
extractor.py
embedder.py
vectordb.py
synthesizers.py
ui.py
models.py
requirements.txt
```

Puedes subirlos desde la interfaz web del Space (**Add file → Upload files**) o usando Git:

```bash
git clone https://huggingface.co/spaces/TU_USUARIO/advanced-ai-research-assistant
cd advanced-ai-research-assistant
cp /ruta/al/repo/*.py .
cp /ruta/al/repo/requirements.txt .
git add .
git commit -m "Initial deployment"
git push
```

---

## Paso 3 — Configurar los Secrets

Ve a tu Space → **Settings → Repository secrets** y agrega:

| Secret | Valor | Obligatorio |
|---|---|---|
| `GROQ_API_KEY` | Tu API key de Groq (gratuita en console.groq.com) | ✅ Sí |
| `HF_TOKEN` | Tu token de Hugging Face (hf.co/settings/tokens) | Opcional |

> **Cómo obtener la Groq API key:**
> 1. Ve a [console.groq.com](https://console.groq.com)
> 2. Crea una cuenta gratuita → **API Keys → Create API Key**
> 3. Copia la key y pégala en el Secret `GROQ_API_KEY`

---

## Paso 4 — Verificar el despliegue

El Space se construye automáticamente (3-5 minutos la primera vez). Cuando el estado cambie a **Running**, accede a la URL. Deberías ver la interfaz con tres pestañas:

- **💬 Consultas** — preguntas sobre los PDFs indexados
- **📄 Mis Archivos** — subir e indexar PDFs
- **⚙️ Configuración** — seleccionar proveedor LLM

---

## Paso 5 — Probar el sistema

1. Ve a **⚙️ Configuración** → selecciona proveedor **anthropic** (usa Groq internamente)
2. Si `GROQ_API_KEY` está como Secret, aparecerá confirmación verde automáticamente
3. Haz clic en **💾 Guardar Cambios**
4. Ve a **📄 Mis Archivos** → sube un PDF académico
5. Ve a **💬 Consultas** → selecciona el documento y escribe tu pregunta

---

## Limitaciones conocidas del tier gratuito

### Groq API — Rate Limiting

El plan gratuito de Groq aplica límites de tokens por minuto (TPM). Mitigaciones aplicadas:
- `max_tokens=600` por llamada (reducido para no saturar el TPM)
- `n_results=3` fragmentos por búsqueda
- Si aparece error de límite, espera 15-30 segundos antes de reintentar

### HF Spaces — Almacenamiento no persistente

ChromaDB se almacena en `/tmp/chroma_db_v32`, que se borra al reiniciar el Space (~48h de inactividad). Los PDFs indexados deben subirse nuevamente tras un reinicio.

### HF Spaces — Sin OAuth en tier gratuito

Sin login OAuth, todos los usuarios comparten el ID `invitado_predeterminado` y ven los mismos documentos. Esta limitación está documentada en la metodología de la tesis.

---

## Solución de problemas frecuentes

| Problema | Solución |
|---|---|
| Space en "Building" por más de 10 min | Revisar logs de construcción; verificar `requirements.txt` |
| `GROQ_API_KEY no encontrada` | Confirmar que el Secret esté en mayúsculas exactas; reiniciar Space |
| Error de `chromadb` al iniciar | Normal en primer inicio; ChromaDB crea su directorio en `/tmp` automáticamente |
| Respuestas cortas o error 429 | Rate limit de Groq activo; esperar 30 segundos e intentar nuevamente |
