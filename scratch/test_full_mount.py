import os
import sys
import django
from starlette.testclient import TestClient

BASE_DIR = os.path.abspath('.')
sys.path.insert(0, os.path.join(BASE_DIR, 'rag_intel_system'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.core.asgi import get_asgi_application
django_app = get_asgi_application()

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
fastapi_app.mount("/api", django_app)

with gr.Blocks() as demo:
    gr.Markdown("Hello")

app = gr.mount_gradio_app(fastapi_app, demo, path="/")
client = TestClient(app)

res_api = client.get("/api/documents/")
print("Test /api/documents/ status:", res_api.status_code)
print("Test /api/documents/ json count:", len(res_api.json()))
res_root = client.get("/")
print("Test / status:", res_root.status_code)
