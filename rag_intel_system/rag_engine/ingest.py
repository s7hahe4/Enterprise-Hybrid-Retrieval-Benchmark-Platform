import re
import os
import concurrent.futures
from pypdf import PdfReader
from documents.models import Document, Chunk, IngestionJob
from .vector_store import add_chunks_to_vector_store
from .bm25_search import get_or_build_bm25_index

def is_heading_candidate(line):
    """
    Detects section headings in text:
    - Lines starting with Markdown headers (#, ##, ###)
    - Short uppercase lines (e.g., 'PROFESSIONAL SUMMARY', 'TECHNICAL SKILLS', 'EDUCATION')
    - Lines ending with a colon or specific structural headers
    """
    s = line.strip()
    if not s or len(s) > 70:
        return False
        
    # Markdown headers
    if re.match(r'^#{1,4}\s+\w+', s):
        return re.sub(r'^#{1,4}\s+', '', s).strip()
        
    # All uppercase words (at least 3 letters, e.g. "TECHNICAL SKILLS")
    clean_alpha = re.sub(r'[^A-Z\s]', '', s).strip()
    if len(clean_alpha) >= 4 and clean_alpha == s and len(s.split()) <= 6:
        return s.title()
        
    # Title Case Header with specific keywords
    header_keywords = [
        'summary', 'skills', 'experience', 'projects', 'education', 
        'research', 'achievements', 'certifications', 'overview',
        'financial', 'results', 'policy', 'benefits', 'compensation'
    ]
    if any(k in s.lower() for k in header_keywords) and len(s.split()) <= 6 and not s.endswith('.'):
        return s.strip()
        
    return False

def detect_and_format_tables(page_text):
    """
    Detects tabular structures by analyzing multi-whitespace column alignments.
    Converts detected tables into clean Markdown tables.
    Returns: (prose_text, list_of_markdown_tables)
    """
    lines = page_text.splitlines()
    prose_lines = []
    table_candidates = []
    current_table = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_table:
                if len(current_table) >= 2:
                    table_candidates.append(current_table)
                else:
                    prose_lines.extend(current_table)
                current_table = []
            continue

        cols = re.split(r'\s{2,}|\t|\|', stripped)
        cols = [c.strip() for c in cols if c.strip()]
        
        if len(cols) >= 2 and len(cols) <= 10:
            current_table.append(cols)
        else:
            if current_table:
                if len(current_table) >= 2:
                    table_candidates.append(current_table)
                else:
                    prose_lines.extend([" ".join(r) for r in current_table])
                current_table = []
            prose_lines.append(stripped)

    if current_table:
        if len(current_table) >= 2:
            table_candidates.append(current_table)
        else:
            prose_lines.extend([" ".join(r) for r in current_table])

    markdown_tables = []
    for raw_table in table_candidates:
        if not raw_table:
            continue
        max_cols = max(len(r) for r in raw_table)
        if max_cols < 2:
            continue
            
        header = raw_table[0]
        while len(header) < max_cols:
            header.append(f"Col {len(header) + 1}")
            
        separator = ["---"] * max_cols
        rows = []
        for r in raw_table[1:]:
            padded = r + [""] * (max_cols - len(r))
            rows.append(padded)
            
        md_table_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |"
        ]
        for row in rows:
            md_table_lines.append("| " + " | ".join(row) + " |")
            
        markdown_tables.append("\n".join(md_table_lines))

    return "\n".join(prose_lines), markdown_tables

def chunk_text_sliding(text, chunk_size=90, overlap=20):
    """Splits text into compact child chunks for high-density vector retrieval."""
    words = text.split()
    chunks = []
    if not words:
        return chunks
        
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
            
    return chunks

def parse_sections(prose_text):
    """
    Parses prose into semantic sections grouped by detected headings.
    Returns: list of dicts [{'heading': str, 'body': str}]
    """
    lines = prose_text.splitlines()
    sections = []
    current_heading = "General"
    current_lines = []
    
    for line in lines:
        heading = is_heading_candidate(line)
        if heading:
            if current_lines:
                body = " ".join(current_lines).strip()
                if body:
                    sections.append({'heading': current_heading, 'body': body})
                current_lines = []
            current_heading = heading
        else:
            if line.strip():
                current_lines.append(line.strip())
                
    if current_lines:
        body = " ".join(current_lines).strip()
        if body:
            sections.append({'heading': current_heading, 'body': body})
            
    return sections

def update_job_progress(job_id, status=None, progress_pct=None, current_step=None, error_message=None, doc=None):
    """Helper to update IngestionJob status and progress atomically in SQLite."""
    if not job_id:
        return
    try:
        job = IngestionJob.objects.filter(id=job_id).first()
        if job:
            if status is not None:
                job.status = status
            if progress_pct is not None:
                job.progress_pct = progress_pct
            if current_step is not None:
                job.current_step = current_step
            if error_message is not None:
                job.error_message = error_message
            if doc is not None:
                job.document = doc
            job.save()
    except Exception:
        pass


def process_and_store_pdf(file_path, filename, job_id=None, version_group=None, version_number=1, access_role='PUBLIC'):
    """
    Advanced Parent-Child & Heading-Aware Ingestion Pipeline with Progress Tracking:
    1. Extracts PDF page by page (15%).
    2. Identifies and formats tabular data into structured Markdown table chunks (35%).
    3. Detects section headings and groups text into cohesive Parent Sections (55%).
    4. Creates Parent Chunks in SQLite (unbroken conceptual context for LLM synthesis).
    5. Splits Parent Chunks into pinpoint Child Chunks (80-100 words) with heading prefixes.
    6. Generates 384d dense vector embeddings (80%).
    7. Indexes Child Chunks and Table Chunks into FAISS and BM25Plus (95%).
    8. Marks document & job completed (100%).
    """
    update_job_progress(job_id, status='extracting', progress_pct=10, current_step='Initializing PDF reader & creating document record...')
    
    doc = Document.objects.create(
        filename=filename, 
        status='processing',
        access_role=access_role or 'PUBLIC',
        version_group=version_group or '',
        version_number=version_number
    )
    update_job_progress(job_id, doc=doc)
    
    try:
        update_job_progress(job_id, status='extracting', progress_pct=20, current_step='Extracting text across document pages...')
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        
        indexable_chunk_ids = []
        text_for_embedding = []
        global_chunk_idx = 0
        
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_page_text = page.extract_text() or ""
            
            # Progress step for multi-page docs
            cur_pct = 20 + int(30 * ((page_idx + 1) / max(total_pages, 1)))
            update_job_progress(job_id, status='tables', progress_pct=min(50, cur_pct), current_step=f'Detecting tables & headings on page {page_num} of {total_pages}...')
            
            prose_text, tables = detect_and_format_tables(raw_page_text)
            
            # 1. Ingest Structured Markdown Tables
            for table_idx, md_table in enumerate(tables):
                table_chunk = Chunk.objects.create(
                    document=doc,
                    parent_chunk=None,
                    heading="Tabular Data",
                    page_number=page_num,
                    text=f"[Structured Table Data - Page {page_num}]\n{md_table}",
                    chunk_index=global_chunk_idx,
                    is_table=True,
                    metadata={'page': page_num, 'table_index': table_idx + 1, 'type': 'markdown_table'}
                )
                global_chunk_idx += 1
                indexable_chunk_ids.append(table_chunk.id)
                text_for_embedding.append(table_chunk.text)
                
            # 2. Section Parsing & Parent-Child Chunking
            sections = parse_sections(prose_text)
            for sec in sections:
                heading = sec['heading']
                body = sec['body']
                
                # Create Parent Chunk (broad unbroken context)
                parent_chunk = Chunk.objects.create(
                    document=doc,
                    parent_chunk=None,
                    heading=heading,
                    page_number=page_num,
                    text=f"[{heading}]\n{body}",
                    chunk_index=global_chunk_idx,
                    is_table=False,
                    metadata={'page': page_num, 'is_parent': True, 'heading': heading}
                )
                global_chunk_idx += 1
                
                # Create Focused Child Chunks (indexed into FAISS & BM25)
                child_passages = chunk_text_sliding(body, chunk_size=90, overlap=20)
                if not child_passages:
                    indexable_chunk_ids.append(parent_chunk.id)
                    text_for_embedding.append(parent_chunk.text)
                else:
                    for c_text in child_passages:
                        formatted_child_text = f"[{heading}] {c_text}"
                        child_chunk = Chunk.objects.create(
                            document=doc,
                            parent_chunk=parent_chunk,
                            heading=heading,
                            page_number=page_num,
                            text=formatted_child_text,
                            chunk_index=global_chunk_idx,
                            is_table=False,
                            metadata={'page': page_num, 'is_child': True, 'parent_id': parent_chunk.id, 'heading': heading}
                        )
                        global_chunk_idx += 1
                        indexable_chunk_ids.append(child_chunk.id)
                        text_for_embedding.append(child_chunk.text)
        
        # Fallback for empty docs
        if not indexable_chunk_ids:
            fallback = Chunk.objects.create(
                document=doc,
                heading="Document Summary",
                page_number=1,
                text=f"Indexed document: {filename}",
                chunk_index=0,
                is_table=False
            )
            indexable_chunk_ids.append(fallback.id)
            text_for_embedding.append(fallback.text)

        # 3. Dense Embeddings & Vector Indexing
        update_job_progress(job_id, status='embedding', progress_pct=75, current_step=f'Generating dense embeddings for {len(text_for_embedding)} chunks...')
        add_chunks_to_vector_store(indexable_chunk_ids, text_for_embedding)

        # 4. Invalidate & update BM25Plus index
        update_job_progress(job_id, status='indexing', progress_pct=92, current_step='Updating BM25Plus sparse index & finalizing metadata...')
        get_or_build_bm25_index(force_rebuild=True)
        
        doc.status = 'completed'
        doc.save()
        
        update_job_progress(job_id, status='completed', progress_pct=100, current_step='Ingestion complete & searchable')
        return doc
        
    except Exception as exc:
        doc.status = 'failed'
        doc.save()
        update_job_progress(job_id, status='failed', progress_pct=0, current_step='Ingestion failed', error_message=str(exc))
        raise exc


# Thread pool background worker for asynchronous ingestion
_ingest_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def submit_async_ingest_job(file_path, filename, job_id, version_group=None, version_number=1, access_role='PUBLIC'):
    """
    Submits a background task to process the uploaded PDF asynchronously,
    allowing the HTTP POST response to return in sub-20ms with a job ID.
    """
    def _worker():
        from django.db import close_old_connections
        close_old_connections()
        try:
            process_and_store_pdf(
                file_path=file_path, 
                filename=filename, 
                job_id=job_id, 
                version_group=version_group, 
                version_number=version_number,
                access_role=access_role
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            update_job_progress(job_id, status='failed', progress_pct=0, current_step='Ingestion failed', error_message=str(exc))
        finally:
            close_old_connections()
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    _ingest_pool.submit(_worker)