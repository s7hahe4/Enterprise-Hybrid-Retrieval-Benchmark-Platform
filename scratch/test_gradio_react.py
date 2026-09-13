import os
import sys
import django
import gradio as gr
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "rag_intel_system")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from django.core.management import call_command
call_command('migrate', interactive=False, verbosity=0)

from django.core.asgi import get_asgi_application
django_app = get_asgi_application()

dist_dir = os.path.join(BASE_DIR, "frontend", "dist")

custom_css = """
body, html, .gradio-container, gradio-app {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    height: 100vh !important;
    overflow: hidden !important;
    background: #0f172a !important;
}
footer { display: none !important; }
.main { padding: 0 !important; }
"""

with gr.Blocks(title="NexusRAG Platform", css=custom_css) as demo:
    gr.HTML('''
    <iframe src="/app" style="position:fixed; top:0; left:0; width:100vw; height:100vh; border:none; margin:0; padding:0; overflow:hidden; z-index:9999;"></iframe>
    ''')

# Mount Django API
demo.app.mount("/api", django_app)

# Mount Frontend Dist Assets
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

print("Gradio + React App + Django ASGI successfully integrated!")
