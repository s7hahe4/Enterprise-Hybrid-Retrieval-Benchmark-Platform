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
def gpu_inference_accelerator(dummy=None):
    """ZeroGPU accelerator hook for Hugging Face hardware allocation"""
    return dummy

import os
import sys
import django
import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

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

# Build FastAPI parent app
fastapi_app = FastAPI(title="NexusRAG Platform API")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount Django ASGI API at /api
fastapi_app.mount("/api", django_app)

# 2. Mount compiled Vite React assets
dist_dir = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(dist_dir):
    assets_dir = os.path.join(dist_dir, "assets")
    if os.path.exists(assets_dir):
        fastapi_app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @fastapi_app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(dist_dir, "index.html"))

    @fastapi_app.get("/favicon.svg")
    async def serve_fav():
        return FileResponse(os.path.join(dist_dir, "favicon.svg"))

    @fastapi_app.get("/icons.svg")
    async def serve_ico():
        return FileResponse(os.path.join(dist_dir, "icons.svg"))

# 3. Create Gradio fallback app and mount at /gradio
with gr.Blocks(title="NexusRAG Architecture Demo") as demo:
    gr.Markdown("# ⚡ NexusRAG Enterprise Platform")
    gr.Markdown("The primary interactive React glassmorphic dashboard is running at the root URL [`/`](/).")

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
