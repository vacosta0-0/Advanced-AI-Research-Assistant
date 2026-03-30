# 🌿 Guía: Dos versiones en el mismo repositorio con Branches

## ¿Qué es un branch (rama)?

Un branch es como una "copia paralela" de tu proyecto dentro del mismo repositorio.
Perfecta para mostrar la evolución de tu tesis:

```
main  ─────────────────────────────────►  versión base (PROYECTO_FINAL.py)
         │
         └── v2-multiproveedor ─────────►  versión avanzada (research_assistant.py)
```

En GitHub se verá así: un solo repositorio con un menú desplegable para cambiar entre versiones.

---

## PASO 1 — Si aún no subiste nada (empezar desde cero)

```bash
# Crear y entrar a la carpeta del proyecto
mkdir advanced-ai-research-assistant
cd advanced-ai-research-assistant

# Inicializar Git
git init

# Crear estructura de carpetas
mkdir -p src data/raw data/processed data/external results models chroma_db chroma_db_v3 tests notebooks
touch data/raw/.gitkeep data/processed/.gitkeep data/external/.gitkeep
touch results/.gitkeep models/.gitkeep chroma_db/.gitkeep chroma_db_v3/.gitkeep
```

---

## PASO 2 — Subir la versión base a `main`

```bash
# Copiar archivos de la versión base
cp /ruta/a/PROYECTO_FINAL.py src/
cp /ruta/a/README.md .           # README de la versión base
cp /ruta/a/.gitignore .
cp /ruta/a/requirements.txt .    # requirements de la versión base

# Confirmar y subir
git add .
git commit -m "feat: versión base del sistema multiagente NLP"

# Conectar con GitHub (reemplaza TU_USUARIO)
git remote add origin https://github.com/TU_USUARIO/advanced-ai-research-assistant.git
git branch -M main
git push -u origin main
```

✅ La versión base ya está en GitHub en la rama `main`.

---

## PASO 3 — Crear la rama `v2-multiproveedor`

```bash
# Crear la nueva rama y cambiar a ella
git checkout -b v2-multiproveedor
```

Ahora reemplaza los archivos con la versión avanzada:

```bash
# Reemplazar el código fuente
cp /ruta/a/research_assistant.py src/

# Reemplazar README y requirements con los de v2
cp /ruta/a/README_v2.md README.md
cp /ruta/a/requirements_v2.txt requirements.txt

# Confirmar los cambios en esta rama
git add .
git commit -m "feat: v3.0 con UI Gradio, procesamiento paralelo y multi-proveedor IA"

# Subir la nueva rama a GitHub
git push -u origin v2-multiproveedor
```

✅ ¡Listo! Ahora tienes dos ramas en el mismo repositorio.

---

## PASO 4 — Verificar en GitHub

1. Ve a `https://github.com/TU_USUARIO/advanced-ai-research-assistant`
2. Busca el menú desplegable que dice **"main"** (arriba a la izquierda)
3. Haz clic y verás las dos ramas:
   - `main` → versión base
   - `v2-multiproveedor` → versión avanzada
4. Al cambiar de rama, cambian todos los archivos automáticamente

---

## Comandos del día a día

### Cambiar entre ramas localmente
```bash
git checkout main               # volver a versión base
git checkout v2-multiproveedor  # ir a versión avanzada
```

### Hacer cambios en una rama específica
```bash
# Primero cambia a la rama que quieres modificar
git checkout v2-multiproveedor

# Haz tus cambios en los archivos...

# Luego confirma y sube
git add .
git commit -m "fix: mejorar detección automática de modelo HF"
git push
```

### Ver diferencias entre las dos versiones
```bash
git diff main v2-multiproveedor -- src/
```

---

## Cómo se verá en tu entregable de tesis

Puedes incluir en tu documento:

```
Repositorio: https://github.com/TU_USUARIO/advanced-ai-research-assistant

Ramas:
- main: Sistema base con arquitectura multiagente y terminal interactiva
- v2-multiproveedor: Evolución con UI web Gradio, procesamiento paralelo
  y soporte para Hugging Face (gratis), Anthropic y OpenAI
```

---

## Errores comunes

**"Already on 'main'"** al crear la rama → Normal, ya estás en main, ejecuta el `checkout -b`

**Cambios sin guardar al cambiar de rama** → Haz `git commit` o `git stash` antes de cambiar

**Push rechazado** → Usa `git push -u origin nombre-de-la-rama` la primera vez
