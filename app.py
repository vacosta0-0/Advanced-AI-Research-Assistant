"""
Advanced AI Research Assistant v3.2 - Edición Multiusuario
Sistema Multiagente con Hugging Face (GRATIS), Anthropic y OpenAI

Este archivo es el punto de entrada principal del sistema.
Organizado de forma modular para facilitar su mantenimiento y explicación.
Incluye soporte para aislamiento de usuarios mediante autenticación.
"""

import os
import sys

def install_dependencies():
    """Asegura que todas las librerías necesarias estén instaladas."""
    import subprocess
    packages = [
        "huggingface_hub>=0.20.0",
        "PyPDF2",
        "pdfminer.six",
        "chromadb",
        "sentence-transformers",
        "gradio>=4.0.0",
        "numpy",
        "requests",
    ]
    print("Instalando dependencias necesarias...")
    for pkg in packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Error al instalar {pkg}: {e}")
    print("Dependencias listas.\n")

# Verificamos si podemos importar las librerías principales; si no, las instalamos.
# Esto se hace ANTES de importar cualquier módulo local que dependa de ellas.
try:
    import gradio as gr
    import PyPDF2
    import chromadb
    import sentence_transformers
    import numpy as np
except ImportError:
    install_dependencies()
    # Una vez instaladas, volvemos a intentar los imports necesarios para el arranque inicial
    import gradio as gr

# IMPORTANTE: Los módulos locales SOLO se importan después de asegurar las dependencias.
# De lo contrario, fallarán al intentar importar gradio, chromadb, etc. en sus respectivos archivos.
from config import CONFIG, DEFAULT_MODEL
from assistant import ResearchAssistant
from ui import build_ui

def main():
    """Función de entrada para iniciar la aplicación con soporte para usuarios."""
    print("=" * 60)
    print("  AI Research Assistant v3.2 - Edición Multiusuario")
    print(f"  Versión de Gradio: {gr.__version__}")
    print("=" * 60)

    # Verificación rápida del token de Hugging Face
    env_tok = os.environ.get("HF_TOKEN", "")
    if env_tok:
        print(f"✅ Token HF detectado (comienza por: {env_tok[:8]}...)")
    else:
        print("⚠️  Aviso: No se encontró HF_TOKEN en las variables de entorno.")

    print(f"\n📌 Modelo por defecto: {DEFAULT_MODEL}")
    print(f"📁 Directorio de la BD: {CONFIG.persist_directory}")
    print("💾 Iniciando sistema con soporte multiusuario...\n")

    # Inicializamos el orquestador y la interfaz de usuario
    assistant = ResearchAssistant(CONFIG)
    demo = build_ui(assistant)

    # Lanzamos la aplicación web. Gradio en Spaces gestiona el login automáticamente
    # si se configura OAuth, de lo contrario Request.username será None.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    main()