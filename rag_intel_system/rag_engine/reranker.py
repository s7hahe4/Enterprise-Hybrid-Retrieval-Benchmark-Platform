import math
import numpy as np

_cross_encoder = None

def get_cross_encoder():
    """Lazily load CrossEncoder to maintain instant server response."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        # ms-marco-MiniLM-L-6-v2 is the industry benchmark for fast, accurate passage re-ranking
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def sigmoid(x):
    """Calibrate raw logits to normalized confidence score [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))

def rerank_chunks(query, candidates, top_k=3):
    """
    Applies Cross-Encoder re-ranking over hybrid candidate chunks.
    Calculates exact semantic matching score for (query, chunk_text) pairs.
    
    Returns:
        tuple (top_k_chunks, full_reranked_candidates_for_mlops)
    """
    if not candidates:
        return [], []
        
    model = get_cross_encoder()
    
    # Construct (query, passage) pairs
    pairs = [(query, c['text']) for c in candidates]
    
    # Compute cross-attention relevance scores
    scores = model.predict(pairs)
    
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
