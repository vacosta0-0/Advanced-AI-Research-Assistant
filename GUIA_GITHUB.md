# 📘 Guía: Cómo subir tu proyecto a GitHub

## Paso 1 — Crear el repositorio en GitHub

1. Ingresa a https://github.com y accede con tu cuenta
2. Haz clic en el botón verde **"New"** (arriba a la izquierda)
3. Configura el repositorio:
   - **Repository name:** `advanced-ai-research-assistant`
   - **Description:** `Sistema multiagente para análisis de artículos académicos con NLP`
   - **Visibility:** Public ✅ (para tu entregable de tesis)
   - **⚠️ NO marques** "Add a README file" (ya tienes uno)
4. Clic en **"Create repository"**

---

## Paso 2 — Preparar la estructura de carpetas en tu computadora

Abre una terminal y ejecuta:

```bash
# Crear carpeta del proyecto
mkdir advanced-ai-research-assistant
cd advanced-ai-research-assistant

# Crear estructura de carpetas
mkdir -p src data/raw data/processed data/external
mkdir -p notebooks results models tests chroma_db
```

Luego copia los archivos en su lugar:
- `PROYECTO_FINAL.py` → dentro de la carpeta `src/`
- `README.md` → en la raíz del proyecto
- `.gitignore` → en la raíz del proyecto
- `requirements.txt` → en la raíz del proyecto

Crea los archivos `.gitkeep` para que Git rastree las carpetas vacías:

```bash
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep
touch results/.gitkeep
touch models/.gitkeep
touch chroma_db/.gitkeep
```

---

## Paso 3 — Inicializar Git y hacer el primer commit

```bash
# Inicializar repositorio Git local
git init

# Agregar todos los archivos
git add .

# Verificar qué archivos se van a subir (opcional)
git status

# Hacer el primer commit
git commit -m "feat: estructura inicial del proyecto y código base del sistema multiagente"
```

---

## Paso 4 — Conectar con GitHub y subir

Copia la URL de tu repositorio desde GitHub (termina en `.git`), luego ejecuta:

```bash
# Conectar repositorio local con GitHub
git remote add origin https://github.com/TU_USUARIO/advanced-ai-research-assistant.git

# Subir los archivos
git branch -M main
git push -u origin main
```

---

## Paso 5 — Verificar en GitHub

1. Ve a `https://github.com/TU_USUARIO/advanced-ai-research-assistant`
2. Deberías ver todos tus archivos y el README renderizado

---

## Comandos útiles para el futuro

Cada vez que hagas cambios en tu código:

```bash
# Ver qué cambió
git status

# Agregar cambios específicos
git add src/PROYECTO_FINAL.py

# O agregar todos los cambios
git add .

# Commit con descripción clara
git commit -m "fix: mejora extracción de texto en PDFs complejos"

# Subir a GitHub
git push
```

### Ejemplos de buenos mensajes de commit para tu tesis:
```
feat: agregar soporte para múltiples idiomas en extracción
fix: corregir error en chunking con documentos grandes  
docs: actualizar README con nuevos resultados
test: agregar pruebas para VectorDatabaseAgent
refactor: optimizar generación de embeddings
```

---

## ¿Problemas comunes?

**Error: `git` no reconocido**  
→ Instala Git desde https://git-scm.com/downloads

**Error de autenticación al hacer push**  
→ GitHub ya no acepta contraseñas. Usa un **Personal Access Token**:
1. GitHub → Settings → Developer settings → Personal access tokens → Generate new token
2. Usa el token como contraseña cuando te lo pida

**Error: `src refspec main does not match any`**  
→ Significa que no tienes commits. Asegúrate de haber ejecutado `git commit` antes del push.
