import math
import os
import re
import numpy as np

_cross_encoder = None

def get_cross_encoder():
    """Lazily load CrossEncoder with cloud auto-detection for memory/CPU constraints."""
    global _cross_encoder
    if _cross_encoder is None:
        if os.environ.get('RENDER') or os.environ.get('LIGHTWEIGHT_MODE', '').lower() == 'true':
            _cross_encoder = "FALLBACK"
            return _cross_encoder

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        try:
            import torch
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"Warning: Could not load CrossEncoder ({e}). Fallback ranking active.")
            _cross_encoder = "FALLBACK"
    return _cross_encoder

def sigmoid(x):
    """Calibrate raw logits to normalized confidence score [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))

def rerank_chunks(query, candidates, top_k=3):
    """
    Applies Cross-Encoder re-ranking over hybrid candidate chunks.
    Calculates exact semantic matching score for (query, chunk_text) pairs.
    """
    if not candidates:
        return [], []
        
    model = get_cross_encoder()
    
    if model == "FALLBACK":
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        for i, c in enumerate(candidates, start=1):
            c_words = set(re.findall(r'\b\w+\b', c.get('text', '').lower()))
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            rrf = float(c.get('rrf_score', 0.5))
            score = round(min(0.99, max(0.40, 0.35 * rrf + 0.65 * overlap)), 4)
            c['final_rank'] = i
            c['rank_delta'] = 0
            c['relevance_score'] = score
            c['raw_rerank_score'] = round(overlap * 2.0, 2)
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        for i, c in enumerate(candidates, start=1):
            c['final_rank'] = i
            c['rank_delta'] = c.get('initial_rrf_rank', i) - i
        return candidates[:top_k], candidates

    # Construct (query, passage) pairs
    pairs = [(query, c['text']) for c in candidates]
    
    # Compute cross-attention relevance scores with memory safety
    try:
        import torch
        with torch.no_grad():
            scores = model.predict(pairs, batch_size=4, show_progress_bar=False)
    except Exception:
        # Fallback on inference failure
        for i, c in enumerate(candidates, start=1):
            c['final_rank'] = i
            c['rank_delta'] = 0
            c['relevance_score'] = round(float(c.get('rrf_score', 0.5)), 4)
            c['raw_rerank_score'] = 1.0
        return candidates[:top_k], candidates
    
    # If single score returned as scalar
    if isinstance(scores, (float, int, np.floating)):
        scores = [float(scores)]
        
    reranked = []
    for candidate, raw_score in zip(candidates, scores):
        calibrated_score = round(sigmoid(float(raw_score)), 4)
        reranked.append({
            **candidate,
            'raw_rerank_score': round(float(raw_score), 4),
            'relevance_score': calibrated_score,
        })
        
    # Sort descending by cross-encoder score
    reranked.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Calculate rank delta (how much Cross-Encoder boosted or demoted each candidate)
    for final_rank, item in enumerate(reranked, start=1):
        item['final_rank'] = final_rank
        item['rank_delta'] = item['initial_rrf_rank'] - final_rank
        
    return reranked[:top_k], reranked
