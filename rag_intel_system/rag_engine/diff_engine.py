import re
import os
from typing import Dict, Any, List
from documents.models import Document, Chunk
from .vector_store import get_embedding_model
from .evaluator import cosine_similarity

def extract_sentences(text: str) -> List[str]:
    """Splits text into meaningful semantic sentences/bullet points for diff comparison."""
    # Split by periods, bullet symbols, or table row breaks
    raw_units = re.split(r'(?<=[.!?])\s+|\n[•\-*]\s+|\n(?=\|)|\n\n+', text)
    cleaned = []
    for u in raw_units:
        s = u.strip().strip('-•* \t')
        if len(s.split()) >= 3 and not s.startswith('| ---'):
            cleaned.append(s)
    return cleaned

def compare_document_versions(doc_id_v1: int, doc_id_v2: int, topic_query: str = "") -> Dict[str, Any]:
    """
    Performs semantic diffing between two document versions:
    1. Extracts passages from both versions (filtered by topic_query if provided).
    2. Identifies added sentences/clauses in v2.
    3. Identifies removed/deprecated sentences/clauses from v1.
    4. Identifies preserved/unchanged terms.
    5. Produces an executive diff summary.
    """
    try:
        doc_v1 = Document.objects.get(pk=doc_id_v1)
        doc_v2 = Document.objects.get(pk=doc_id_v2)
    except Document.DoesNotExist:
        return {'error': 'One or both document versions not found.'}

    # Retrieve chunks for both documents
    chunks_v1 = list(Chunk.objects.filter(document=doc_v1).order_by('chunk_index'))
    chunks_v2 = list(Chunk.objects.filter(document=doc_v2).order_by('chunk_index'))

    if not chunks_v1 or not chunks_v2:
        return {
            'error': 'One of the document versions has no indexed chunks.',
            'doc_v1': doc_v1.filename,
            'doc_v2': doc_v2.filename
        }

    # Aggregate text for each document (or filter by topic_query if specified)
    model = get_embedding_model()

    if topic_query.strip():
        q_vec = model.encode([topic_query])[0]

        # Score chunks by relevance to topic
        def filter_top_chunks(chunks):
            scored = []
            for c in chunks:
                c_vec = model.encode([c.text[:500]])[0]
                sim = cosine_similarity(q_vec, c_vec)
                scored.append((sim, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for sim, c in scored[:5]]

        eval_chunks_v1 = filter_top_chunks(chunks_v1)
        eval_chunks_v2 = filter_top_chunks(chunks_v2)
    else:
        eval_chunks_v1 = chunks_v1[:8]
        eval_chunks_v2 = chunks_v2[:8]

    text_v1 = " ".join([c.text for c in eval_chunks_v1])
    text_v2 = " ".join([c.text for c in eval_chunks_v2])

    units_v1 = extract_sentences(text_v1)
    units_v2 = extract_sentences(text_v2)

    if not units_v1 or not units_v2:
        return {'error': 'Could not extract sufficient text for comparison.'}

    # Encode sentence units to find cosine alignments
    vecs_v1 = model.encode(units_v1)
    vecs_v2 = model.encode(units_v2)

    SIMILARITY_THRESHOLD = 0.85
    MODIFIED_THRESHOLD = 0.65

    added = []
    removed = []
    modified = []
    unchanged = []

    # Check each unit in v2 against v1 to detect additions and modifications
    for idx2, (u2, v2) in enumerate(zip(units_v2, vecs_v2)):
        best_sim = -1.0
        best_match_idx = -1
        for idx1, v1 in enumerate(vecs_v1):
            sim = cosine_similarity(v2, v1)
            if sim > best_sim:
                best_sim = sim
                best_match_idx = idx1

        if best_sim >= SIMILARITY_THRESHOLD:
            unchanged.append({
                'clause': u2,
                'similarity': round(best_sim, 3)
            })
        elif best_sim >= MODIFIED_THRESHOLD:
            modified.append({
                'v1_original': units_v1[best_match_idx],
                'v2_updated': u2,
                'similarity': round(best_sim, 3)
            })
        else:
            added.append({
                'clause': u2,
                'status': 'added_in_v2'
            })

    # Check units in v1 that have no close match in v2 to detect removals
    for idx1, (u1, v1) in enumerate(zip(units_v1, vecs_v1)):
        best_sim = max([cosine_similarity(v1, v2) for v2 in vecs_v2]) if len(vecs_v2) > 0 else 0.0
        if best_sim < MODIFIED_THRESHOLD:
            removed.append({
                'clause': u1,
                'status': 'removed_from_v1'
            })

    # Generate Executive Summary
    summary_diff = (
        f"Compared '{doc_v1.filename}' (v{doc_v1.version_number}) with '{doc_v2.filename}' (v{doc_v2.version_number}). "
        f"Identified {len(added)} newly added clauses, {len(modified)} modified clauses, "
        f"{len(removed)} removed clauses, and {len(unchanged)} unchanged provisions."
    )

    # If OpenAI API is available, generate natural language executive summary
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if api_key and not api_key.startswith('sk-placeholder'):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                f"You are an enterprise document auditor. Compare these two versions:\n"
                f"DOCUMENT 1 ({doc_v1.filename}):\n{text_v1[:1200]}\n\n"
                f"DOCUMENT 2 ({doc_v2.filename}):\n{text_v2[:1200]}\n\n"
                f"Topic Focus: {topic_query or 'General Comparison'}\n\n"
                "Provide a concise 3-4 bullet point executive diff detailing what was added, what was changed, and what was removed."
            )
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.2
            )
            ai_summary = completion.choices[0].message.content.strip()
            if ai_summary:
                summary_diff = ai_summary
        except Exception:
            pass

    return {
        'doc_v1': {
            'id': doc_v1.id,
            'filename': doc_v1.filename,
            'version': doc_v1.version_number,
            'version_group': doc_v1.version_group
        },
        'doc_v2': {
            'id': doc_v2.id,
            'filename': doc_v2.filename,
            'version': doc_v2.version_number,
            'version_group': doc_v2.version_group
        },
        'topic_query': topic_query,
        'summary_diff': summary_diff,
        'added_clauses': added[:10],
        'modified_clauses': modified[:10],
        'removed_clauses': removed[:10],
        'unchanged_count': len(unchanged),
        'metrics': {
            'added_count': len(added),
            'modified_count': len(modified),
            'removed_count': len(removed),
            'unchanged_count': len(unchanged)
        }
    }
