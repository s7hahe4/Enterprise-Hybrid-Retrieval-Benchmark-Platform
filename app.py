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

import os
import sys
import json
import pandas as pd

# Add backend directory to sys.path and setup Django
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "rag_intel_system")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

import django
django.setup()

from django.core.management import call_command
call_command('migrate', interactive=False, verbosity=0)

import gradio as gr
from rag_engine.rag_stream import stream_rag_pipeline
from rag_engine.ingest import process_and_store_pdf
from rag_engine.benchmark import run_benchmark_suite, CONFIGURATIONS, generate_synthetic_benchmark
from documents.models import Document, Chunk

# Pre-populate sample documents if database is empty
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
                        filename=fname,
                        file=fpath,
                        access_role=role
                    )
                    process_and_store_pdf(doc.id, fpath)
                except Exception as e:
                    print(f"Error seeding {fname}: {e}")

# -------------------------------------------------------------
# Core Execution Handlers
# -------------------------------------------------------------

def stream_rag_chat(message, history, user_role):
    if not message or not message.strip():
        yield history, "⚡ Enter a query above to see real-time pipeline telemetry.", pd.DataFrame()
        return

    chat_history = []
    for user_msg, assistant_msg in history:
        chat_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            chat_history.append({"role": "assistant", "content": assistant_msg})

    new_history = history + [[message, ""]]
    
    stream_gen = stream_rag_pipeline(
        query=message,
        conversation_history=chat_history,
        user_role=user_role
    )

    full_answer = ""
    telemetry_html = f"""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 14px; margin-bottom: 12px; backdrop-filter: blur(8px);">
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
            <span style="background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.4);">
                SECURITY CLEARANCE: {user_role}
            </span>
            <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.4);">
                STATUS: PROCESSING
            </span>
        </div>
        <p style="color: #94a3b8; font-size: 12px; margin: 0;">Executing agentic intent classification, hybrid retrieval & Cross-Encoder re-ranking...</p>
    </div>
    """
    matrix_df = pd.DataFrame(columns=["Rank", "Document", "RRF Score", "Cross-Encoder Relevance", "Shift"])

    for event in stream_gen:
        evt_type = event.get("event")
        evt_data = event.get("data", {})

        if evt_type == "status":
            step = evt_data.get("step", "Processing")
            telemetry_html = f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 14px; margin-bottom: 12px; backdrop-filter: blur(8px);">
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
                    <span style="background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.4);">
                        CLEARANCE: {user_role}
                    </span>
                    <span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; border: 1px solid rgba(59, 130, 246, 0.4);">
                        STEP: {step.upper()}
                    </span>
                </div>
                <p style="color: #cbd5e1; font-size: 12px; margin: 0;">{step}</p>
            </div>
            """
            yield new_history, telemetry_html, matrix_df

        elif evt_type == "token":
            full_answer += evt_data.get("token", "")
            new_history[-1][1] = full_answer
            yield new_history, telemetry_html, matrix_df

        elif evt_type == "rerank_matrix":
            rows = []
            for item in evt_data:
                init_rank = item.get("initial_rank", "-")
                final_rank = item.get("final_rank", "-")
                shift = f"#{init_rank} → #{final_rank}"
                rows.append({
                    "Rank": f"#{final_rank}",
                    "Document": item.get("document", "Doc"),
                    "RRF Score": f"{item.get('initial_rrf_score', 0):.4f}",
                    "Cross-Encoder Relevance": f"{item.get('cross_encoder_score', 0):.4f}",
                    "Shift": shift
                })
            matrix_df = pd.DataFrame(rows) if rows else matrix_df
            yield new_history, telemetry_html, matrix_df

        elif evt_type == "telemetry":
            intent = evt_data.get("intent", "IN_DOMAIN_RAG")
            latency = evt_data.get("latency_ms", 0)
            chunks_count = evt_data.get("chunks_retrieved", 0)
            rewritten = evt_data.get("rewritten_query", message)
            
            telemetry_html = f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 14px; margin-bottom: 12px; backdrop-filter: blur(8px);">
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;">
                    <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.4);">
                        INTENT: {intent}
                    </span>
                    <span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700; border: 1px solid rgba(168, 85, 247, 0.4);">
                        LATENCY: {latency} ms
                    </span>
                    <span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700; border: 1px solid rgba(59, 130, 246, 0.4);">
                        CHUNKS: {chunks_count}
                    </span>
                </div>
                <div style="font-size: 11px; color: #94a3b8; line-height: 1.5;">
                    <span style="color: #64748b;">Contextual Rewriter:</span> <span style="color: #f1f5f9; font-style: italic;">"{rewritten}"</span>
                </div>
            </div>
            """
            yield new_history, telemetry_html, matrix_df


def handle_document_upload(file, role):
    if not file:
        return "⚠️ Please choose a PDF file to upload.", get_document_list()
    
    file_path = file.name if hasattr(file, 'name') else str(file)
    filename = os.path.basename(file_path)
    
    try:
        doc = Document.objects.create(
            filename=filename,
            file=file_path,
            access_role=role
        )
        chunk_count = process_and_store_pdf(doc.id, file_path)
        return f"✅ Successfully ingested '{filename}' into RBAC tier '{role}' ({chunk_count} chunks indexed).", get_document_list()
    except Exception as e:
        return f"❌ Upload failed: {str(e)}", get_document_list()


def get_document_list():
    docs = Document.objects.all().order_by('-upload_date')
    data = []
    for d in docs:
        data.append({
            "ID": d.id,
            "Filename": d.filename,
            "Security Role": d.access_role,
            "Indexed Chunks": d.chunks.count(),
            "Version": f"v{d.version_number}"
        })
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["ID", "Filename", "Security Role", "Indexed Chunks", "Version"])


@spaces.GPU(duration=120)
def run_benchmark(num_questions):
    try:
        dataset = generate_synthetic_benchmark(num_questions=int(num_questions))
        result = run_benchmark_suite(test_cases=dataset, configurations=CONFIGURATIONS)
        summary = result.get("summary_metrics", [])
        
        rows = []
        for s in summary:
            rows.append({
                "Configuration": s["name"],
                "Recall@1": f"{s['recall@1']*100:.1f}%",
                "Recall@3": f"{s['recall@3']*100:.1f}%",
                "Recall@5": f"{s['recall@5']*100:.1f}%",
                "MRR": f"{s['mrr']:.3f}",
                "NDCG@5": f"{s['ndcg@5']:.3f}",
                "Mean Latency": f"{s['mean_latency_ms']} ms",
                "MRR Lift": f"+{s['mrr_lift_pct']:.1f}%" if s['mrr_lift_pct'] > 0 else "Baseline"
            })
        return pd.DataFrame(rows), f"🏆 Benchmark evaluation completed over {len(dataset)} empirical test queries across 5 architectures."
    except Exception as e:
        return pd.DataFrame(), f"Benchmark error: {str(e)}"


# -------------------------------------------------------------
# Ultra-Premium Dark Glassmorphic Theme & CSS
# -------------------------------------------------------------

custom_theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]
).set(
    body_background_fill="#090d16",
    body_background_fill_dark="#090d16",
    body_text_color="#f8fafc",
    body_text_color_dark="#f8fafc",
    block_background_fill="rgba(15, 23, 42, 0.75)",
    block_background_fill_dark="rgba(15, 23, 42, 0.75)",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_color_dark="rgba(255, 255, 255, 0.08)",
    block_radius="16px",
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="rgba(99, 102, 241, 0.5)",
    input_background_fill="rgba(2, 6, 23, 0.7)",
    input_background_fill_dark="rgba(2, 6, 23, 0.7)",
    input_border_color="rgba(255, 255, 255, 0.12)",
    input_border_color_dark="rgba(255, 255, 255, 0.12)",
    input_radius="12px"
)

custom_css = """
/* Background Glow */
body, .gradio-container {
    background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.12) 0%, transparent 40%),
                #070b14 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Glassmorphic Cards */
.glass-panel {
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
}

/* Header Styling */
.hero-title {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    letter-spacing: -0.02em;
}

.hero-sub {
    color: #94a3b8 !important;
    font-size: 0.92rem !important;
    margin-top: 6px !important;
    line-height: 1.4 !important;
}

/* Tech Pills */
.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.03em;
    background: rgba(99, 102, 241, 0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

.dot-green {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
}

/* Tab Headers */
.tab-nav button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    color: #94a3b8 !important;
}

.tab-nav button.selected {
    background: rgba(99, 102, 241, 0.25) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.3) !important;
}

/* Chatbot Styles */
.chatbot .message.user {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border-radius: 14px 14px 2px 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.chatbot .message.bot {
    background: rgba(15, 23, 42, 0.85) !important;
    backdrop-filter: blur(8px) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px 14px 14px 2px !important;
}

/* Tables */
table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

th {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}

td {
    padding: 10px 14px !important;
    font-size: 12px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
}

/* Quick Question Trigger Chips */
.quick-chip {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    color: #cbd5e1 !important;
    transition: all 0.2s ease !important;
}

.quick-chip:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}
"""

# -------------------------------------------------------------
# Gradio Layout Construction
# -------------------------------------------------------------

with gr.Blocks(title="NexusRAG Platform") as demo:
    # Header Bar
    with gr.Row():
        with gr.Column(scale=8):
            gr.HTML("""
            <div>
                <h1 class="hero-title">🏛️ NexusRAG: Enterprise Hybrid Retrieval & Benchmark Platform</h1>
                <p class="hero-sub">Production Hybrid Search (FAISS + BM25) • Cross-Encoder Re-Ranking • Document-Level RBAC • Empirical RAGAS MLOps</p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                    <span class="pill-badge"><span class="dot-green"></span> Live Cloud Deployment</span>
                    <span class="pill-badge">⚡ ZeroGPU A10G Accelerated</span>
                    <span class="pill-badge">🧠 ms-marco-MiniLM Re-Ranker</span>
                    <span class="pill-badge">🛡️ Role-Based Access Control</span>
                </div>
            </div>
            """)
        with gr.Column(scale=4, min_width=200):
            role_selector = gr.Dropdown(
                choices=["PUBLIC", "ENGINEERING", "HR", "FINANCE", "ADMIN"],
                value="PUBLIC",
                label="Active Security Clearance (RBAC)",
                info="Switch role to test pre-retrieval boundaries"
            )

    gr.HTML("<div style='height: 8px;'></div>")

    # Main Tabs
    with gr.Tabs():
        # TAB 1: RAG COPILOT & GUARDRAILS
        with gr.TabItem("💬 RAG Copilot & Guardrails"):
            with gr.Row():
                # Left: Chat Area
                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(
                        height=520,
                        placeholder="Welcome to NexusRAG! Ask any question or click a preset below to see hybrid retrieval and RBAC in action.",
                        show_label=False
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask a question about engineering, HR, or finance policies...",
                            show_label=False,
                            scale=9,
                            container=False
                        )
                        send_btn = gr.Button("⚡ Send", variant="primary", scale=2)

                    # Quick Presets
                    gr.HTML("<p style='font-size: 11px; color: #64748b; margin: 6px 0 2px 0; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;'>⚡ Quick Test Prompts:</p>")
                    with gr.Row():
                        p1 = gr.Button("🏢 Engineering Architecture", size="sm", elem_classes=["quick-chip"])
                        p2 = gr.Button("🔒 Executive Bonus (RBAC Test)", size="sm", elem_classes=["quick-chip"])
                        p3 = gr.Button("📈 Q4 Financial Performance", size="sm", elem_classes=["quick-chip"])
                        p4 = gr.Button("🛡️ Zero-Trust Security", size="sm", elem_classes=["quick-chip"])

                # Right: Telemetry & Explainability
                with gr.Column(scale=5):
                    gr.HTML("<p style='font-size: 12px; font-weight: 700; color: #cbd5e1; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em;'>🧠 Real-Time MLOps Telemetry</p>")
                    telemetry_panel = gr.HTML(
                        """
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 18px; text-align: center; color: #64748b; font-size: 12px;">
                            ⚡ Send a query to see real-time pipeline latency, intent routing, and contextual query rewrite.
                        </div>
                        """
                    )
                    
                    gr.HTML("<p style='font-size: 12px; font-weight: 700; color: #cbd5e1; margin: 12px 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em;'>🎯 Cross-Encoder Re-Ranking Matrix (ms-marco-MiniLM)</p>")
                    matrix_table = gr.DataFrame(
                        headers=["Rank", "Document", "RRF Score", "Cross-Encoder Relevance", "Shift"],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True
                    )

            # Wiring Chat
            chat_inputs = [msg_input, chatbot, role_selector]
            chat_outputs = [chatbot, telemetry_panel, matrix_table]

            send_btn.click(stream_rag_chat, inputs=chat_inputs, outputs=chat_outputs).then(
                lambda: "", None, msg_input
            )
            msg_input.submit(stream_rag_chat, inputs=chat_inputs, outputs=chat_outputs).then(
                lambda: "", None, msg_input
            )

            # Preset wiring
            p1.click(lambda: "What are the core principles of our engineering architecture?", None, msg_input)
            p2.click(lambda: "What is the executive bonus percentage policy?", None, msg_input)
            p3.click(lambda: "Summarize the Q4 financial performance and net margin", None, msg_input)
            p4.click(lambda: "Explain the Zero-Trust network boundary isolation model", None, msg_input)

        # TAB 2: DOCUMENT INGESTION
        with gr.TabItem("📄 Knowledge Base & Ingestion"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.HTML("<h3 style='color: #f1f5f9; font-size: 1rem; margin-top: 0;'>Upload Custom PDF Document</h3>")
                    pdf_input = gr.File(label="Choose PDF File", file_types=[".pdf"])
                    upload_role = gr.Dropdown(
                        choices=["PUBLIC", "ENGINEERING", "HR", "FINANCE", "ADMIN"],
                        value="PUBLIC",
                        label="Assign Security Clearance Tier"
                    )
                    upload_btn = gr.Button("📥 Ingest & Index Document", variant="primary")
                    upload_status = gr.HTML("<p style='color: #64748b; font-size: 12px; margin-top: 6px;'>Ready to ingest into FAISS + BM25 indices.</p>")
                
                with gr.Column(scale=7):
                    gr.HTML("<h3 style='color: #f1f5f9; font-size: 1rem; margin-top: 0;'>Indexed Enterprise Documents</h3>")
                    doc_table = gr.DataFrame(
                        value=get_document_list(),
                        interactive=False,
                        wrap=True
                    )
                    refresh_docs_btn = gr.Button("🔄 Refresh Document Index", size="sm")
                    refresh_docs_btn.click(get_document_list, None, doc_table)

            upload_btn.click(
                handle_document_upload,
                inputs=[pdf_input, upload_role],
                outputs=[upload_status, doc_table]
            )

        # TAB 3: BENCHMARK LAB
        with gr.TabItem("📊 Empirical Benchmark Lab"):
            gr.HTML("""
            <div style="margin-bottom: 12px;">
                <h3 style="color: #f1f5f9; font-size: 1.1rem; margin: 0 0 4px 0;">Empirical IR Retrieval Evaluation Suite</h3>
                <p style="color: #94a3b8; font-size: 12px; margin: 0;">Quantifies retrieval quality across 5 architectures: FAISS (Dense), BM25 (Sparse), RRF Hybrid, Hybrid + Cross-Encoder, and NexusRAG.</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    benchmark_slider = gr.Slider(
                        minimum=3, maximum=15, value=5, step=1,
                        label="Evaluation Dataset Size (Test Queries)"
                    )
                    benchmark_btn = gr.Button("🚀 Run Empirical IR Benchmark Suite", variant="primary")
                    benchmark_status = gr.Markdown("Click above to run the 5-architecture evaluation suite.")
                
                with gr.Column(scale=9):
                    benchmark_table = gr.DataFrame(
                        headers=["Configuration", "Recall@1", "Recall@3", "Recall@5", "MRR", "NDCG@5", "Mean Latency", "MRR Lift"],
                        interactive=False,
                        wrap=True
                    )

            benchmark_btn.click(
                run_benchmark,
                inputs=[benchmark_slider],
                outputs=[benchmark_table, benchmark_status]
            )

# Mount Django ASGI REST & Streaming API on FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

fastapi_app = FastAPI(title="NexusRAG API")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from django.core.asgi import get_asgi_application
django_app = get_asgi_application()
fastapi_app.mount("/api", django_app)

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
