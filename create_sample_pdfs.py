import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

output_dir = r"d:\rag_project\sample_test_docs"
os.makedirs(output_dir, exist_ok=True)

styles = getSampleStyleSheet()
title_style = styles['Title']
heading_style = styles['Heading2']
body_style = styles['BodyText']

# -------------------------------------------------------------
# Document 1: Engineering Architecture Handbook (Engineering / Public)
# -------------------------------------------------------------
doc1_path = os.path.join(output_dir, "engineering_architecture_handbook.pdf")
doc1 = SimpleDocTemplate(doc1_path, pagesize=letter)
story1 = [
    Paragraph("Engineering Architecture & Service Guidelines", title_style),
    Spacer(1, 14),
    Paragraph("Microservice Infrastructure Overview", heading_style),
    Paragraph(
        "Our core platform relies on Python 3.11 with Django REST Framework and FastAPI for sub-millisecond asynchronous execution. "
        "Asynchronous workloads such as vector indexing, embedding calculations, and background PDF ingestion are orchestrated "
        "using Celery workers coupled with Redis as a message broker. All microservices are containerized using multi-stage Docker images "
        "and fronted by an Nginx reverse proxy.",
        body_style
    ),
    Spacer(1, 14),
    Paragraph("Vector Database & Embedding Specifications", heading_style),
    Paragraph(
        "For dense semantic retrieval, the platform implements FAISS CPU IndexIDMap utilizing 384-dimensional dense vectors "
        "generated via sentence-transformers (all-MiniLM-L6-v2). Lexical keyword retrieval is handled concurrently by BM25Plus, "
        "which eliminates the zero-IDF saturation artifact on small corporate corpora. Both dense and sparse ranks are merged "
        "using Reciprocal Rank Fusion (RRF) with constant k=60, before passing the top 20 candidates to the Cross-Encoder re-ranker.",
        body_style
    ),
    Spacer(1, 14),
    Paragraph("Service Latency and Availability SLAs", heading_style),
]

table1_data = [
    ["Service Name", "Target TTFT", "P95 Latency", "Availability SLA"],
    ["RAG Streaming SSE", "45 ms", "120 ms", "99.95%"],
    ["Hybrid Vector Retrieval", "12 ms", "28 ms", "99.99%"],
    ["Cross-Encoder Reranker", "35 ms", "70 ms", "99.90%"],
    ["Background PDF Ingestion", "N/A (Async)", "8.5 sec", "99.50%"]
]
t1 = Table(table1_data, colWidths=[140, 90, 90, 110])
t1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
]))
story1.append(t1)
doc1.build(story1)
print(f"Generated: {doc1_path}")

# -------------------------------------------------------------
# Document 2: HR Compensation & Benefits Policy v1 (HR Restricted)
# -------------------------------------------------------------
doc2_path = os.path.join(output_dir, "hr_compensation_policy_v1.pdf")
doc2 = SimpleDocTemplate(doc2_path, pagesize=letter)
story2 = [
    Paragraph("Corporate Compensation & Employee Benefits Policy (v1)", title_style),
    Spacer(1, 14),
    Paragraph("Confidential HR Directive - Version 1.0", ParagraphStyle('Sub', parent=body_style, textColor=colors.HexColor("#b91c1c"))),
    Spacer(1, 14),
    Paragraph("Workplace Schedule and Remote Work Policy", heading_style),
    Paragraph(
        "All full-time staff members must work from the designated corporate office a minimum of 3 days per week. "
        "Employees are permitted to work remotely up to 2 days per week upon manager approval. Core working hours are 9:00 AM to 5:00 PM EST.",
        body_style
    ),
    Spacer(1, 14),
    Paragraph("Executive Bonus Structure and Equity Grants", heading_style),
    Paragraph(
        "Senior engineering leads and directors are eligible for an annual performance bonus targeted at 15% of their base compensation, "
        "contingent upon achieving company ARR targets. Equity stock options (RSUs) vest on a standard 4-year schedule with a 1-year cliff.",
        body_style
    ),
    Spacer(1, 14),
    Paragraph("Retirement Matching and Health Benefits", heading_style),
    Paragraph(
        "The company provides a 100% 401(k) retirement match up to 4% of eligible annual salary. Comprehensive medical, dental, "
        "and vision health insurance premiums are 85% employer-subsidized for employees and 70% for dependents.",
        body_style
    )
]
doc2.build(story2)
print(f"Generated: {doc2_path}")

# -------------------------------------------------------------
# Document 3: HR Compensation & Benefits Policy v2 (HR Restricted - For Diff Testing)
# -------------------------------------------------------------
doc3_path = os.path.join(output_dir, "hr_compensation_policy_v2.pdf")
doc3 = SimpleDocTemplate(doc3_path, pagesize=letter)
story3 = [
    Paragraph("Corporate Compensation & Employee Benefits Policy (v2)", title_style),
    Spacer(1, 14),
    Paragraph("Confidential HR Directive - Version 2.0 (Updated Amendments)", ParagraphStyle('Sub', parent=body_style, textColor=colors.HexColor("#047857"))),
    Spacer(1, 14),
    Paragraph("Workplace Schedule and Remote Work Policy", heading_style),
    Paragraph(
        "All full-time staff members are permitted to work remotely up to 4 days per week under our hybrid-first initiative. "
        "Employees must work from the physical office 1 day per week for cross-functional collaboration. Core working hours are 10:00 AM to 4:00 PM EST.",
        body_style
    ),
    Spacer(1, 14),
    Paragraph("Executive Bonus Structure and Equity Grants", heading_style),
    Paragraph(
        "Senior engineering leads and directors are eligible for an increased annual performance bonus targeted at 22% of their base compensation, "
        "plus an accelerated quarterly milestone incentive. Equity stock options (RSUs) now vest quarterly after a 1-year cliff.",
        body_style
    ),
    Spacer(1, 14),
    Paragraph("Retirement Matching and Health Benefits", heading_style),
    Paragraph(
        "The company has expanded its 401(k) retirement match to 100% match up to 6% of eligible annual salary. "
        "Comprehensive medical, dental, and vision insurance premiums are now 100% employer-covered for all employees.",
        body_style
    ),
    Spacer(1, 14),
    Paragraph("New Home Office Equipment Stipend", heading_style),
    Paragraph(
        "Under the revised policy, all hybrid and remote team members receive an annual tax-free $1,200 home office hardware stipend "
        "for ergonomic chairs, standing desks, and ultra-wide monitor displays.",
        body_style
    )
]
doc3.build(story3)
print(f"Generated: {doc3_path}")

# -------------------------------------------------------------
# Document 4: Financial Quarterly Audit Report (Finance Restricted)
# -------------------------------------------------------------
doc4_path = os.path.join(output_dir, "q4_financial_audit_report.pdf")
doc4 = SimpleDocTemplate(doc4_path, pagesize=letter)
story4 = [
    Paragraph("Q4 Audited Financial Performance & Revenue Statement", title_style),
    Spacer(1, 14),
    Paragraph("Executive Financial Summary", heading_style),
    Paragraph(
        "During the fourth fiscal quarter of 2025, total enterprise Annual Recurring Revenue (ARR) reached $42.8M, "
        "representing a year-over-year expansion rate of 48%. Gross profit margins improved to 76.4%, driven by cloud infrastructure "
        "optimizations and self-hosted vector retrieval caches.",
        body_style
    ),
    Spacer(1, 14),
    Paragraph("Quarterly Revenue and Operating Expenditure Ledger", heading_style),
]

table4_data = [
    ["Fiscal Quarter", "ARR ($M)", "Gross Margin", "Net Burn ($M)", "EBITDA ($M)"],
    ["Q1 2025", "$28.5M", "68.2%", "$2.4M", "-$1.8M"],
    ["Q2 2025", "$32.1M", "71.0%", "$1.9M", "-$1.1M"],
    ["Q3 2025", "$37.4M", "74.5%", "$1.2M", "-$0.3M"],
    ["Q4 2025", "$42.8M", "76.4%", "$0.7M", "+$1.4M"]
]
t4 = Table(table4_data, colWidths=[90, 80, 85, 90, 85])
t4.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#064e3b")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#10b981")),
]))
story4.append(t4)
doc4.build(story4)
print(f"Generated: {doc4_path}")
