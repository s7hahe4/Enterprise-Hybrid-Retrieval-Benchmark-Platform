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
# Ensure database tables exist
call_command('migrate', interactive=False, verbosity=0)

import gradio as gr
from rag_engine.rag_stream import stream_rag_pipeline
from rag_engine.ingest import process_and_store_pdf
from rag_engine.benchmark import run_benchmark_suite, CONFIGURATIONS, generate_synthetic_benchmark
from rag_engine.diff_engine import compare_document_versions
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
                    process_and_store_pdf(fpath, fname, access_role=role)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")

# -------------------------------------------------------------
# Gradio Tab Handlers
# -------------------------------------------------------------
def chat_stream(message, history, user_role):
    """Streams RAG responses with conversational memory, RBAC, and telemetry."""
    if not message or not message.strip():
        yield history, "No query provided.", "", ""

    history_payload = []
    if history:
        for turn in history[-6:]:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                history_payload.append({"role": "user", "content": turn[0]})
                history_payload.append({"role": "assistant", "content": turn[1] or ""})

    current_history = list(history or [])
    current_history.append([message, ""])

    full_answer = ""
    intent_info = {}
    security_info = {}
    metrics_info = {}
    rerank_info = {}

    for raw_event in stream_rag_pipeline(message, conversation_history=history_payload, user_role=user_role):
        if raw_event.startswith("data: "):
            try:
                payload = json.loads(raw_event[6:].strip())
                event_type = payload.get("type")
                data = payload.get("payload", {})

                if event_type == "token":
                    full_answer += data.get("delta", "")
                    current_history[-1][1] = full_answer
                    yield current_history, "Streaming...", "", ""
                elif event_type == "intent":
                    intent_info = data
                elif event_type == "security":
                    security_info = data
                elif event_type == "rerank":
                    rerank_info = data
                elif event_type == "metrics":
                    metrics_info = data
            except Exception:
                pass

    # Format telemetry output
    telemetry_md = f"### 🧠 MLOps & Telemetry\n"
    telemetry_md += f"- **Active Role:** `{user_role}`\n"
    if security_info:
        telemetry_md += f"- **Pre-Retrieval Filter:** {security_info.get('blocked_chunks_count', 0)} restricted chunks blocked\n"
    if intent_info:
        telemetry_md += f"- **Intent Classified:** `{intent_info.get('intent')}` (Confidence: {int(intent_info.get('confidence', 0)*100)}%)\n"
    if metrics_info:
        ragas = metrics_info.get("ragas", {})
        latency = metrics_info.get("latency_breakdown", {})
        telemetry_md += f"- **RAGAS Faithfulness:** {ragas.get('faithfulness', 0)*100:.1f}%\n"
        telemetry_md += f"- **RAGAS Relevancy:** {ragas.get('relevancy', 0)*100:.1f}%\n"
        telemetry_md += f"- **Total Latency:** {latency.get('total_ms', 0)} ms\n"

    # Re-ranking Matrix Table
    matrix_df = pd.DataFrame(columns=["Rank", "Document", "Initial RRF", "Relevance", "Δ Rank"])
    if rerank_info and "top_chunks" in rerank_info:
        rows = []
        for c in rerank_info["top_chunks"]:
            delta = c.get("rank_delta", 0)
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            rows.append({
                "Rank": c.get("final_rank", 1),
                "Document": c.get("document_filename", ""),
                "Initial RRF": c.get("initial_rank", 1),
                "Relevance": f"{c.get('relevance_score', 0)*100:.1f}%",
                "Δ Rank": delta_str
            })
        if rows:
            matrix_df = pd.DataFrame(rows)

    yield current_history, telemetry_md, matrix_df, full_answer


def upload_document(file_obj, access_role):
    if not file_obj:
        return "No file provided.", get_doc_list()
    try:
        filename = os.path.basename(file_obj.name)
        doc = process_and_store_pdf(file_obj.name, filename, access_role=access_role)
        return f"✅ Successfully ingested '{filename}' [{access_role} tier] with {doc.chunks.count()} chunks.", get_doc_list()
    except Exception as e:
        return f"❌ Upload error: {str(e)}", get_doc_list()


def get_doc_list():
    docs = Document.objects.all().order_by("-upload_date")
    data = []
    for d in docs:
        data.append({
            "ID": d.id,
            "Filename": d.filename,
            "Security Role": d.access_role,
            "Chunks": d.chunks.count(),
            "Version": f"v{d.version_number}"
        })
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["ID", "Filename", "Security Role", "Chunks", "Version"])


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
        return pd.DataFrame(rows), f"🏆 Benchmark completed over {len(dataset)} test queries across 5 architectures."
    except Exception as e:
        return pd.DataFrame(), f"Benchmark error: {str(e)}"


# -------------------------------------------------------------
# Gradio Blocks UI Layout
# -------------------------------------------------------------
custom_css = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }
.header-badge { font-size: 0.75rem; padding: 3px 8px; border-radius: 9999px; background: #6366f1; color: #fff; font-weight: 600; }
"""

with gr.Blocks(title="NexusRAG | Enterprise Hybrid RAG Platform") as demo:
    gr.Markdown(
        """
        # 🏛️ NexusRAG: Enterprise Hybrid Retrieval & Benchmark Platform
        **Production-Grade RAG Architecture with Cross-Encoder Re-Ranking, Document-Level RBAC & Real-Time RAGAS MLOps**
        """
    )

    with gr.Tabs():
        # TAB 1: RAG Copilot
        with gr.Tab("💬 RAG Copilot & Guardrails"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="NexusRAG Assistant", height=480)
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Ask a question about the indexed knowledge base...", label="Your Question", scale=4)
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    gr.Markdown("**⚡ Test Pipeline Guardrails & Scenarios:**")
                    with gr.Row():
                        p1 = gr.Button("ML & Backend Specs")
                        p2 = gr.Button("HR Payroll Test (RBAC)")
                        p3 = gr.Button("Chitchat Bypass")
                        p4 = gr.Button("OOD Cookie Recipe")

                with gr.Column(scale=2):
                    user_role = gr.Dropdown(
                        choices=["PUBLIC", "ENGINEERING", "HR", "FINANCE", "ADMIN"],
                        value="PUBLIC",
                        label="🔐 Active Security Role (RBAC)",
                        info="Switch role tier to test pre-retrieval security boundaries"
                    )
                    telemetry_box = gr.Markdown("### 🧠 MLOps & Telemetry\n*Send a message to see real-time metrics.*")
                    matrix_table = gr.DataFrame(label="Cross-Encoder Re-Ranking Matrix (ms-marco-MiniLM)", headers=["Rank", "Document", "Initial RRF", "Relevance", "Δ Rank"])

            # Wire up chat handlers
            msg.submit(chat_stream, [msg, chatbot, user_role], [chatbot, telemetry_box, matrix_table, msg])
            send_btn.click(chat_stream, [msg, chatbot, user_role], [chatbot, telemetry_box, matrix_table, msg])
            
            p1.click(lambda: "What machine learning and backend engineering experience does Shahedul have?", None, msg)
            p2.click(lambda: "What is the confidential payroll ledger and executive bonus package?", None, msg)
            p3.click(lambda: "Hello! Who are you and how can you help me today?", None, msg)
            p4.click(lambda: "What is the best recipe for baking chocolate fudge cookies?", None, msg)

        # TAB 2: Document Ingestion
        with gr.Tab("📄 Document Ingestion"):
            with gr.Row():
                with gr.Column():
                    upload_file = gr.File(label="Upload PDF Document", file_types=[".pdf"])
                    upload_role = gr.Dropdown(choices=["PUBLIC", "ENGINEERING", "HR", "FINANCE", "ADMIN"], value="PUBLIC", label="Assign Security Tier")
                    upload_btn = gr.Button("Ingest & Index Document", variant="primary")
                    upload_status = gr.Textbox(label="Ingestion Status", interactive=False)
                with gr.Column():
                    gr.Markdown("### 📚 Indexed Knowledge Base")
                    doc_table = gr.DataFrame(value=get_doc_list, label="Documents in Vector Store", every=5)

            upload_btn.click(upload_document, [upload_file, upload_role], [upload_status, doc_table])

        # TAB 3: Benchmark Lab
        with gr.Tab("📊 Benchmark Lab"):
            gr.Markdown(
                """
                ### 🧪 Automated IR Retrieval Benchmark Suite
                Evaluates **5 retrieval architectures** across classical Information Retrieval metrics: **Recall@1, Recall@3, Recall@5, MRR (Mean Reciprocal Rank), and NDCG@5**.
                """
            )
            with gr.Row():
                num_q = gr.Slider(minimum=2, maximum=10, value=4, step=1, label="Number of Benchmark Questions")
                bench_btn = gr.Button("🚀 Run Benchmark Suite", variant="primary")
            
            bench_status = gr.Markdown("*Click 'Run Benchmark Suite' to execute evaluation.*")
            bench_results = gr.DataFrame(label="Empirical Retrieval Leaderboard")

            bench_btn.click(run_benchmark, [num_q], [bench_results, bench_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
