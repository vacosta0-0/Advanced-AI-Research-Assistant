import os
import gradio as gr
from config import DEFAULT_MODEL, HF_WORKING_MODELS

def build_ui(assistant):
    """Construye y configura la interfaz de usuario con Gradio y soporte multiusuario."""

    env_token = os.environ.get("HF_TOKEN", "")
    token_status_html = (
        '<div style="background:#064e3b;border:1px solid #059669;border-radius:8px;'
        'padding:10px 14px;margin:8px 0">'
        '<b style="color:#34d399">✅ Token HF detectado en Secrets</b>'
        '<span style="color:#6ee7b7;font-size:0.9rem"> — listo para usar</span>'
        '</div>'
        if env_token else
        '<div style="background:#7f1d1d;border:1px solid #dc2626;border-radius:8px;'
        'padding:10px 14px;margin:8px 0">'
        '<b style="color:#fca5a5">⚠️ Token HF no encontrado</b>'
        '<span style="color:#fca5a5;font-size:0.9rem"> — escríbelo en Configuración</span>'
        '</div>'
    )

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_user_id(request: gr.Request):
        """Obtiene un identificador de usuario desde la sesión de Gradio."""
        if request and request.username:
            return request.username
        return "invitado_predeterminado"

    def save_config(provider, hf_token, hf_model, ant_key, oai_key):
        """Guarda la configuración global del sistema (compartida por ahora)."""
        assistant.config.provider = provider
        assistant._synthesizer = None

        if provider == "huggingface":
            tok = (hf_token or "").strip() or os.environ.get("HF_TOKEN", "")
            if not tok:
                return "⚠️ No hay token de Hugging Face."
            assistant.config.hf_token = tok
            assistant.config.hf_model = hf_model or DEFAULT_MODEL
            return f"✅ Hugging Face configurado con {assistant.config.hf_model}"

        elif provider == "anthropic":
            if not ant_key.strip():
                return "⚠️ Ingresa tu API key de Anthropic."
            assistant.config.anthropic_api_key = ant_key.strip()
            return f"✅ Anthropic configurado con {assistant.config.claude_model}"

        elif provider == "openai":
            if not oai_key.strip():
                return "⚠️ Ingresa tu API key de OpenAI."
            assistant.config.openai_api_key = oai_key.strip()
            return f"✅ OpenAI configurado con {assistant.config.openai_model}"

    def toggle_panels(provider):
        """Alterna paneles de configuración."""
        return (
            gr.update(visible=(provider == "huggingface")),
            gr.update(visible=(provider == "anthropic")),
            gr.update(visible=(provider == "openai")),
        )

    def upload_pdfs(files, request: gr.Request, progress=gr.Progress()):
        """Sube y procesa PDFs aislados por usuario."""
        if not files:
            return "⚠️ No se seleccionaron archivos."

        user_id = get_user_id(request)
        paths = [f.name for f in files]
        progress(0, desc=f"Procesando PDFs para {user_id}...")
        msg = assistant.add_documents(paths, user_id=user_id)
        progress(1)
        return msg

    def ask(question, history, selected_docs, request: gr.Request):
        """Genera respuesta filtrando por el contexto del usuario actual."""
        if not question.strip():
            return history, ""

        user_id = get_user_id(request)
        try:
            answer = assistant.query(question, user_id=user_id, doc_filter=selected_docs if selected_docs else None)
        except Exception as e:
            answer = f"❌ Error: {str(e)}"

        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    def refresh_docs(request: gr.Request):
        """Actualiza la lista de documentos del usuario."""
        user_id = get_user_id(request)
        docs = assistant.list_documents(user_id)
        return "\n".join(f"- {d}" for d in docs) if docs else "*(No tienes documentos subidos)*"

    def refresh_checklist(request: gr.Request):
        """Actualiza el checklist para el usuario actual."""
        user_id = get_user_id(request)
        docs = assistant.list_documents(user_id)
        return gr.update(choices=docs, value=docs)

    def select_all_docs(request: gr.Request):
        user_id = get_user_id(request)
        docs = assistant.list_documents(user_id)
        return gr.update(choices=docs, value=docs)

    def deselect_all_docs(request: gr.Request):
        user_id = get_user_id(request)
        docs = assistant.list_documents(user_id)
        return gr.update(choices=docs, value=[])

    def clear_db(request: gr.Request):
        """Limpia solo la BD del usuario actual."""
        user_id = get_user_id(request)
        msg = assistant.clear(user_id)
        return msg, "*(Tus documentos han sido eliminados)*", gr.update(choices=[], value=[])

    def upd_claude(model_name):
        assistant.config.claude_model = model_name
        assistant._synthesizer = None

    def upd_oai(model_name):
        assistant.config.openai_model = model_name
        assistant._synthesizer = None

    # ── layout ────────────────────────────────────────────────────────────────

    with gr.Blocks(title="AI Research Assistant v3.2", fill_height=True) as demo:

        gr.HTML("""
        <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;
                    border-radius:12px;padding:20px 30px;margin-bottom:15px">
            <h1 style="color:#f1f5f9;font-size:1.6rem;font-weight:700;margin:0">
                🔬 AI Research Assistant
            </h1>
            <p style="color:#94a3b8;font-size:0.85rem">
                Análisis multi-documento con aislamiento de usuario.
            </p>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: Consultas ─────────────────────────────────────────────
            with gr.Tab("💬 Consultas"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=240):
                        gr.Markdown("### 📂 Mis Documentos")
                        doc_checklist = gr.CheckboxGroup(
                            choices=[], value=[], label="",
                            info="Selecciona en qué documentos buscar."
                        )
                        with gr.Row():
                            select_all_btn   = gr.Button("Todos",   size="sm")
                            deselect_all_btn = gr.Button("Ninguno", size="sm")
                        refresh_btn_tab3 = gr.Button("🔄 Actualizar lista", variant="secondary", size="sm")

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=450,
                            show_label=False,
                            avatar_images=("👤", "🤖"),
                        )
                        with gr.Row():
                            question_input = gr.Textbox(
                                placeholder="Escribe tu pregunta de investigación...",
                                label="", scale=5, container=False,
                            )
                            send_btn = gr.Button("Enviar", scale=1, variant="primary")
                        gr.Examples(
                            examples=[
                                ["¿Cuáles son las conclusiones principales?"],
                                ["¿Qué metodologías se comparan entre los artículos?"],
                                ["Resume los hallazgos comunes en estos documentos."],
                            ],
                            inputs=question_input,
                        )

            # ── TAB 2: Gestión de Archivos ───────────────────────────────────
            with gr.Tab("📄 Mis Archivos"):
                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            label="Sube tus archivos PDF (se guardarán solo en tu cuenta)",
                            file_types=[".pdf"],
                            file_count="multiple",
                        )
                        upload_btn = gr.Button("⬆️ Indexar en mi cuenta", variant="primary")
                        upload_status = gr.Textbox(label="Resultado", interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Documentos indexados")
                        doc_list = gr.Markdown("*(No tienes documentos subidos)*")
                        clear_btn   = gr.Button("🗑️ Eliminar mis datos", variant="stop")
                        clear_status = gr.Textbox(label="", interactive=False)

            # ── TAB 3: Configuración ─────────────────────────────────────────
            with gr.Tab("⚙️ Configuración"):
                with gr.Row():
                    with gr.Column(scale=2):
                        provider_radio = gr.Radio(
                            choices=["huggingface", "anthropic", "openai"],
                            value="huggingface",
                            label="Proveedor de Modelos LLM",
                        )
                        with gr.Group(visible=True) as panel_hf:
                            gr.HTML(token_status_html)
                            hf_token_input = gr.Textbox(
                                label="Token de Hugging Face",
                                type="password",
                                placeholder="Escribe tu token si no está en Secrets...",
                            )
                            hf_model_dropdown = gr.Dropdown(
                                choices=list(HF_WORKING_MODELS.values()),
                                value=DEFAULT_MODEL,
                                label="Modelo de Lenguaje",
                            )
                        with gr.Group(visible=False) as panel_ant:
                            ant_key_input = gr.Textbox(
                                label="Anthropic API Key", type="password",
                            )
                            claude_radio = gr.Radio(
                                choices=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                                value="claude-sonnet-4-6", label="Versión de Claude",
                            )
                            claude_radio.change(upd_claude, inputs=claude_radio)
                        with gr.Group(visible=False) as panel_oai:
                            oai_key_input = gr.Textbox(
                                label="OpenAI API Key", type="password",
                            )
                            oai_model_radio = gr.Radio(
                                choices=["gpt-4o-mini", "gpt-4o"],
                                value="gpt-4o-mini", label="Versión de GPT",
                            )
                            oai_model_radio.change(upd_oai, inputs=oai_model_radio)

                        save_btn = gr.Button("💾 Guardar Cambios", variant="primary")
                        config_status = gr.Textbox(label="Estado de la configuración", interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Nota Multiusuario
                        Cada usuario que inicie sesión verá **únicamente sus propios documentos**.
                        La configuración de modelos es global para esta instancia del asistente.
                        """)

        # ── Eventos ──────────────────────────────────────────────────────────
        # Usamos gr.LoginButton si estamos en Spaces para forzar autenticación
        # Para entornos locales, funcionará como "invitado_predeterminado"

        upload_btn.click(upload_pdfs, inputs=file_upload, outputs=upload_status).then(
            refresh_docs, outputs=doc_list).then(
            refresh_checklist, outputs=doc_checklist)

        refresh_btn_tab3.click(refresh_checklist, outputs=doc_checklist)
        clear_btn.click(clear_db, outputs=[clear_status, doc_list, doc_checklist])

        select_all_btn.click(select_all_docs,   outputs=doc_checklist)
        deselect_all_btn.click(deselect_all_docs, outputs=doc_checklist)

        send_btn.click(
            ask,
            inputs=[question_input, chatbot, doc_checklist],
            outputs=[chatbot, question_input],
        )
        question_input.submit(
            ask,
            inputs=[question_input, chatbot, doc_checklist],
            outputs=[chatbot, question_input],
        )

        provider_radio.change(toggle_panels, inputs=provider_radio,
                              outputs=[panel_hf, panel_ant, panel_oai])
        save_btn.click(
            save_config,
            inputs=[provider_radio, hf_token_input, hf_model_dropdown, ant_key_input, oai_key_input],
            outputs=config_status,
        )

        # Al cargar la página, intentamos refrescar las listas
        demo.load(refresh_docs, outputs=doc_list)
        demo.load(refresh_checklist, outputs=doc_checklist)

    return demo