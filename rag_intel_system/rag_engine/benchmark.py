import time
import math
import re
import random
import os
from typing import List, Dict, Any, Set, Tuple

from documents.models import Chunk, BenchmarkRun
from .vector_store import dense_vector_search
from .bm25_search import bm25_search
from .hybrid_retriever import hybrid_retrieve_candidates
from .reranker import rerank_chunks
from .evaluator import evaluate_ragas_metrics

CONFIGURATIONS = [
    'FAISS_ONLY',
    'BM25_ONLY',
    'NAIVE_HYBRID',
    'HYBRID_RRF',
    'HYBRID_RRF_CROSS_ENCODER'
]

CONFIG_DISPLAY_NAMES = {
    'FAISS_ONLY': 'FAISS Only (Dense Vector)',
    'BM25_ONLY': 'BM25 Only (Sparse Lexical)',
    'NAIVE_HYBRID': 'Naive Hybrid (Linear Combination)',
    'HYBRID_RRF': 'Hybrid + RRF (Reciprocal Rank Fusion)',
    'HYBRID_RRF_CROSS_ENCODER': 'Hybrid + RRF + Cross-Encoder (Full Pipeline)'
}

# --- METRIC COMPUTATIONS ---

def compute_recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Proportion of relevant chunks retrieved in top k."""
    if not relevant_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    return len(top_k_set.intersection(relevant_ids)) / float(len(relevant_ids))

def compute_hit_rate_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """1.0 if at least one relevant chunk appears in top k, else 0.0."""
    if not relevant_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    return 1.0 if len(top_k_set.intersection(relevant_ids)) > 0 else 0.0

def compute_precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Fraction of top k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    return len(top_k_set.intersection(relevant_ids)) / float(k)

def compute_mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant chunk."""
    for idx, cid in enumerate(retrieved_ids):
        if cid in relevant_ids:
            return 1.0 / (idx + 1)
    return 0.0

def compute_ndcg_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k (binary relevance)."""
    if not relevant_ids or k <= 0:
        return 0.0
    
    dcg = 0.0
    for idx, cid in enumerate(retrieved_ids[:k]):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(idx + 2)
            
    idcg = 0.0
    for idx in range(min(k, len(relevant_ids))):
        idcg += 1.0 / math.log2(idx + 2)
        
    return (dcg / idcg) if idcg > 0 else 0.0


# --- CONFIGURATION RETRIEVAL RUNNERS ---

def retrieve_faiss_only(query: str, top_k: int = 20) -> List[int]:
    results = dense_vector_search(query, top_k=top_k)
    return [item['chunk_id'] for item in results]

def retrieve_bm25_only(query: str, top_k: int = 20) -> List[int]:
    results = bm25_search(query, top_k=top_k)
    return [item['chunk_id'] for item in results]

def retrieve_naive_hybrid(query: str, top_k: int = 20) -> List[int]:
    """
    Standard naive baseline: min-max normalizes FAISS similarities and BM25 scores,
    averaging them linearly (0.5 * dense + 0.5 * sparse) without Reciprocal Rank Fusion.
    """
    dense_results = dense_vector_search(query, top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)

    dense_map = {item['chunk_id']: item['similarity'] for item in dense_results}
    bm25_map = {item['chunk_id']: item['score'] for item in bm25_results}

    all_cids = set(dense_map.keys()).union(set(bm25_map.keys()))
    if not all_cids:
        return []

    max_bm25 = max(bm25_map.values()) if bm25_map else 1.0
    min_bm25 = min(bm25_map.values()) if bm25_map else 0.0
    bm25_range = max(max_bm25 - min_bm25, 0.0001)

    combined_scores = {}
    for cid in all_cids:
        d_score = dense_map.get(cid, 0.0)
        raw_b = bm25_map.get(cid, 0.0)
        norm_b = (raw_b - min_bm25) / bm25_range if bm25_map else 0.0
        combined_scores[cid] = 0.5 * d_score + 0.5 * norm_b

    sorted_pairs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in sorted_pairs[:top_k]]

def retrieve_hybrid_rrf(query: str, top_k: int = 20) -> List[int]:
    candidates = hybrid_retrieve_candidates(query, top_k=top_k)
    return [c['chunk_id'] for c in candidates]

def retrieve_hybrid_rrf_cross_encoder(query: str, top_k: int = 20) -> List[int]:
    candidates = hybrid_retrieve_candidates(query, top_k=max(top_k * 2, 15))
    if not candidates:
        return []
    top_k_reranked, _ = rerank_chunks(query, candidates, top_k=top_k)
    return [c['chunk_id'] for c in top_k_reranked]


RETRIEVAL_HANDLERS = {
    'FAISS_ONLY': retrieve_faiss_only,
    'BM25_ONLY': retrieve_bm25_only,
    'NAIVE_HYBRID': retrieve_naive_hybrid,
    'HYBRID_RRF': retrieve_hybrid_rrf,
    'HYBRID_RRF_CROSS_ENCODER': retrieve_hybrid_rrf_cross_encoder
}


# --- SYNTHETIC TEST-SUITE GENERATOR ---

def generate_synthetic_benchmark(num_questions: int = 8) -> List[Dict[str, Any]]:
    """
    Generates a deterministic or LLM-synthesized evaluation test suite
    directly from indexed document chunks.
    """
    # Prioritize child chunks (which are indexed in FAISS and BM25) or fallback to all
    chunks = list(Chunk.objects.filter(parent_chunk__isnull=False).select_related('document'))
    if not chunks:
        chunks = list(Chunk.objects.all().select_related('document'))

    if not chunks:
        return []

    test_cases = []
    seen_queries = set()

    # If OPENAI_API_KEY is available in environment, use LLM for high-fidelity synthesis
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if api_key and not api_key.startswith('sk-placeholder'):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            sample_chunks = random.sample(chunks, min(len(chunks), num_questions * 2))
            for chunk in sample_chunks:
                if len(test_cases) >= num_questions:
                    break
                text_snippet = chunk.text[:400]
                prompt = (
                    f"Given the following passage from '{chunk.document.filename}' "
                    f"(Section: '{chunk.heading}'):\n"
                    f"\"\"\"{text_snippet}\"\"\"\n\n"
                    "Generate a single, clear, specific search query that a researcher or recruiter would ask, "
                    "where this exact passage provides the primary answer. Return ONLY the query string, nothing else."
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60,
                    temperature=0.3
                )
                q_text = response.choices[0].message.content.strip().strip('"')
                if q_text and q_text not in seen_queries:
                    seen_queries.add(q_text)
                    test_cases.append({
                        'id': len(test_cases) + 1,
                        'query': q_text,
                        'relevant_chunk_ids': [chunk.id],
                        'target_heading': chunk.heading or 'General Content',
                        'document_filename': chunk.document.filename,
                        'passage_excerpt': text_snippet[:150] + '...'
                    })
            if test_cases:
                return test_cases
        except Exception:
            # Fall through to heuristic extraction if OpenAI call fails
            pass

    # Heuristic question synthesis from salient entities, headings, and key sentences
    clean_words = lambda t: re.findall(r'\b[A-Za-z0-9_\-\.]{3,}\b', t)
    
    # Shuffle or step through chunks
    step = max(1, len(chunks) // num_questions)
    sampled = chunks[::step][:num_questions * 2]

    for chunk in sampled:
        if len(test_cases) >= num_questions:
            break
            
        text = chunk.text.strip()
        lines = [ln.strip() for ln in text.split('\n') if ln.strip() and not ln.startswith('|')]
        heading = chunk.heading.strip() if chunk.heading else ""

        query = None
        # Pattern 1: Heading-based targeted question
        if heading and len(heading.split()) <= 6:
            heading_title = heading.title()
            queries = [
                f"What are the details regarding {heading_title}?",
                f"What skills or qualifications are listed under {heading_title}?",
                f"What experience or projects are described in {heading_title}?"
            ]
            query = random.choice(queries)

        # Pattern 2: Extract technical skills or proper nouns from text
        if not query or query in seen_queries:
            sentences = re.split(r'[\.\n•\-]\s+', text)
            salient_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5 and len(s.strip().split()) <= 25]
            if salient_sentences:
                best_s = salient_sentences[0]
                tokens = clean_words(best_s)
                meaningful_tokens = [w for w in tokens if w.lower() not in {'this', 'that', 'with', 'from', 'have', 'were', 'been', 'which', 'using', 'based'}]
                if len(meaningful_tokens) >= 3:
                    keyword_phrase = " ".join(meaningful_tokens[:3])
                    query = f"What details or achievements relate to {keyword_phrase}?"

        # Pattern 3: Fallback first phrase inquiry
        if not query or query in seen_queries:
            words = clean_words(text)
            if len(words) >= 4:
                query = f"Information about {' '.join(words[:4])} in {chunk.document.filename}"

        if query and query not in seen_queries:
            seen_queries.add(query)
            test_cases.append({
                'id': len(test_cases) + 1,
                'query': query,
                'relevant_chunk_ids': [chunk.id],
                'target_heading': chunk.heading or 'General Section',
                'document_filename': chunk.document.filename,
                'passage_excerpt': text[:160] + ('...' if len(text) > 160 else '')
            })

    return test_cases


# --- BENCHMARK EXECUTION ENGINE ---

def run_benchmark_suite(
    test_cases: List[Dict[str, Any]] = None,
    configurations: List[str] = None,
    num_synthetic_questions: int = 8,
    k_eval: int = 5,
    save_run: bool = True,
    run_name: str = None
) -> Dict[str, Any]:
    """
    Executes an empirical RAG benchmark across selected retrieval configurations.
    Returns comparative evaluation metrics and detailed per-query rank breakdowns.
    """
    if configurations is None:
        configurations = CONFIGURATIONS.copy()

    # 1. Resolve or generate test cases
    if not test_cases:
        test_cases = generate_synthetic_benchmark(num_questions=num_synthetic_questions)

    if not test_cases:
        return {
            'error': 'No test cases available. Please upload and index documents first.',
            'summary_metrics': {},
            'detailed_results': []
        }

    # 2. Initialize tracking data structures
    config_metrics = {
        cfg: {
            'display_name': CONFIG_DISPLAY_NAMES.get(cfg, cfg),
            'recall_1': [],
            'recall_3': [],
            'recall_5': [],
            'hit_rate_3': [],
            'hit_rate_5': [],
            'mrr': [],
            'ndcg_5': [],
            'precision_3': [],
            'latencies_ms': []
        }
        for cfg in configurations
    }

    per_query_results = []

    # 3. Run evaluation per query across all configurations
    for tc in test_cases:
        q_id = tc.get('id', len(per_query_results) + 1)
        query = tc['query']
        relevant_ids = set(tc.get('relevant_chunk_ids', []))

        query_breakdown = {
            'id': q_id,
            'query': query,
            'target_heading': tc.get('target_heading', ''),
            'document_filename': tc.get('document_filename', ''),
            'relevant_chunk_ids': list(relevant_ids),
            'passage_excerpt': tc.get('passage_excerpt', ''),
            'config_evaluations': {}
        }

        for cfg in configurations:
            handler = RETRIEVAL_HANDLERS.get(cfg)
            if not handler:
                continue

            # Measure retrieval latency
            t0 = time.perf_counter()
            retrieved_ids = handler(query, top_k=k_eval * 3)
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)

            # Compute individual IR metrics
            r1 = compute_recall_at_k(retrieved_ids, relevant_ids, k=1)
            r3 = compute_recall_at_k(retrieved_ids, relevant_ids, k=3)
            r5 = compute_recall_at_k(retrieved_ids, relevant_ids, k=5)
            h3 = compute_hit_rate_at_k(retrieved_ids, relevant_ids, k=3)
            h5 = compute_hit_rate_at_k(retrieved_ids, relevant_ids, k=5)
            mrr_val = compute_mrr(retrieved_ids, relevant_ids)
            ndcg_val = compute_ndcg_at_k(retrieved_ids, relevant_ids, k=5)
            p3 = compute_precision_at_k(retrieved_ids, relevant_ids, k=3)

            # Find rank of first hit
            first_hit_rank = None
            for idx, cid in enumerate(retrieved_ids):
                if cid in relevant_ids:
                    first_hit_rank = idx + 1
                    break

            # Record
            config_metrics[cfg]['recall_1'].append(r1)
            config_metrics[cfg]['recall_3'].append(r3)
            config_metrics[cfg]['recall_5'].append(r5)
            config_metrics[cfg]['hit_rate_3'].append(h3)
            config_metrics[cfg]['hit_rate_5'].append(h5)
            config_metrics[cfg]['mrr'].append(mrr_val)
            config_metrics[cfg]['ndcg_5'].append(ndcg_val)
            config_metrics[cfg]['precision_3'].append(p3)
            config_metrics[cfg]['latencies_ms'].append(duration_ms)

            query_breakdown['config_evaluations'][cfg] = {
                'retrieved_chunk_ids': retrieved_ids[:k_eval],
                'first_hit_rank': first_hit_rank,
                'hit_within_k': (first_hit_rank is not None and first_hit_rank <= k_eval),
                'mrr': round(mrr_val, 4),
                'recall_3': round(r3, 4),
                'recall_5': round(r5, 4),
                'ndcg_5': round(ndcg_val, 4),
                'latency_ms': duration_ms
            }

        per_query_results.append(query_breakdown)

    # 4. Compute Aggregate Leaderboard Summary
    summary = {}
    faiss_mrr = 0.0
    faiss_ndcg = 0.0

    avg = lambda lst: (sum(lst) / len(lst)) if lst else 0.0

    for cfg in configurations:
        metrics = config_metrics[cfg]
        mean_mrr = avg(metrics['mrr'])
        mean_ndcg = avg(metrics['ndcg_5'])
        mean_r1 = avg(metrics['recall_1'])
        mean_r3 = avg(metrics['recall_3'])
        mean_r5 = avg(metrics['recall_5'])
        mean_h3 = avg(metrics['hit_rate_3'])
        mean_h5 = avg(metrics['hit_rate_5'])
        mean_p3 = avg(metrics['precision_3'])
        mean_lat = avg(metrics['latencies_ms'])

        if cfg == 'FAISS_ONLY':
            faiss_mrr = mean_mrr
            faiss_ndcg = mean_ndcg

        summary[cfg] = {
            'config_id': cfg,
            'display_name': metrics['display_name'],
            'mean_mrr': round(mean_mrr, 4),
            'mean_ndcg_5': round(mean_ndcg, 4),
            'recall_1': round(mean_r1, 4),
            'recall_3': round(mean_r3, 4),
            'recall_5': round(mean_r5, 4),
            'hit_rate_3': round(mean_h3, 4),
            'hit_rate_5': round(mean_h5, 4),
            'precision_3': round(mean_p3, 4),
            'avg_latency_ms': round(mean_lat, 2),
            'mrr_lift_pct': 0.0,
            'ndcg_lift_pct': 0.0
        }

    # Calculate lift over FAISS baseline
    for cfg in configurations:
        if faiss_mrr > 0:
            lift_mrr = ((summary[cfg]['mean_mrr'] - faiss_mrr) / faiss_mrr) * 100
            summary[cfg]['mrr_lift_pct'] = round(lift_mrr, 1)
        if faiss_ndcg > 0:
            lift_ndcg = ((summary[cfg]['mean_ndcg_5'] - faiss_ndcg) / faiss_ndcg) * 100
            summary[cfg]['ndcg_lift_pct'] = round(lift_ndcg, 1)

    # 5. Persist to Database if requested
    benchmark_run_id = None
    if save_run:
        name = run_name or f"Benchmark Experiment ({len(test_cases)} queries)"
        db_run = BenchmarkRun.objects.create(
            name=name,
            total_queries=len(test_cases),
            configurations=configurations,
            summary_metrics=summary,
            detailed_results=per_query_results
        )
        benchmark_run_id = db_run.id

    return {
        'benchmark_run_id': benchmark_run_id,
        'name': run_name or f"Benchmark Experiment ({len(test_cases)} queries)",
        'total_queries': len(test_cases),
        'configurations_tested': configurations,
        'summary_metrics': summary,
        'detailed_results': per_query_results,
        'test_cases': test_cases
    }
