try:
    import spaces
except Exception:
    class _MockSpaces:
        def GPU(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(f):
                return f
            return decorator
    spaces = _MockSpaces()

@spaces.GPU
def gpu_init(x=""):
    """ZeroGPU hook to satisfy Hugging Face startup check"""
    return x

import os
import sys
import django
import gradio as gr
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Setup Django backend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "rag_intel_system")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from django.core.management import call_command
call_command('migrate', interactive=False, verbosity=0)

from django.core.asgi import get_asgi_application
django_app = get_asgi_application()

from documents.models import Document
from rag_engine.ingest import process_and_store_pdf

# Pre-populate sample documents if empty
if Document.objects.count() == 0:
    sample_docs_dir = os.path.join(BASE_DIR, "sample_test_docs")
    if os.path.exists(sample_docs_dir):
        pdf_files = [
            ("engineering_architecture_handbook.pdf", "ENGINEERING"),
            ("hr_compensation_policy_v1.pdf", "HR"),
            ("hr_compensation_policy_v2.pdf", "HR"),
            ("q4_financial_audit_report.pdf", "FINANCE")
        ]
        for fname, role in pdf_files:
            fpath = os.path.join(sample_docs_dir, fname)
            if os.path.exists(fpath):
                try:
                    doc = Document.objects.create(
                        title=fname.replace("_", " ").replace(".pdf", "").title(),
                        filename=fname,
                        file=fpath,
                        access_role=role
                    )
                    process_and_store_pdf(doc.id, fpath)
                except Exception as e:
                    print(f"Error seeding {fname}: {e}")

dist_dir = os.path.join(BASE_DIR, "frontend", "dist")

custom_css = """
body, html, .gradio-container, gradio-app {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    overflow: hidden !important;
    background: #0f172a !important;
}
footer { display: none !important; }
.main { padding: 0 !important; }
#react-app-frame {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    border: none;
    margin: 0;
    padding: 0;
    overflow: hidden;
    z-index: 999999;
}
"""

with gr.Blocks(title="NexusRAG Platform") as demo:
    # ZeroGPU event wire
    init_btn = gr.Button("Init", visible=False)
    init_out = gr.Textbox(visible=False)
    init_btn.click(fn=gpu_init, inputs=[init_out], outputs=[init_out])

    # Native React 19 Glassmorphic Interface Embed
    gr.HTML('<iframe id="react-app-frame" src="/app"></iframe>')

# Mount Django ASGI API at /api
demo.app.mount("/api", django_app)

# Mount Frontend Build Assets
if os.path.exists(dist_dir):
    assets_dir = os.path.join(dist_dir, "assets")
    if os.path.exists(assets_dir):
        demo.app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @demo.app.get("/app")
    async def serve_app():
        return FileResponse(os.path.join(dist_dir, "index.html"))

    @demo.app.get("/favicon.svg")
    async def serve_fav():
        return FileResponse(os.path.join(dist_dir, "favicon.svg"))

    @demo.app.get("/icons.svg")
    async def serve_ico():
        return FileResponse(os.path.join(dist_dir, "icons.svg"))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)
