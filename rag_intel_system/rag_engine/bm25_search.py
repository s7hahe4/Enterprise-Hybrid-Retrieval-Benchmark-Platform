import re
from rank_bm25 import BM25Plus
from documents.models import Chunk

_bm25_index = None
_indexed_chunk_ids = []

def tokenize_corpus(text):
    """Clean alphanumeric tokenization for BM25 ranking."""
    return re.findall(r'\b\w+\b', text.lower())

def get_or_build_bm25_index(force_rebuild=False):
    """
    Maintains an in-memory BM25Plus index of all chunks currently in SQLite.
    BM25Plus provides superior lower-bound frequency penalization and prevents
    the zero-IDF artifact on small document counts.
    """
    global _bm25_index, _indexed_chunk_ids
    
    current_ids = list(Chunk.objects.values_list('id', flat=True))
    if not force_rebuild and _bm25_index is not None and current_ids == _indexed_chunk_ids:
        return _bm25_index, _indexed_chunk_ids
        
    chunks = list(Chunk.objects.all().values('id', 'text'))
    if not chunks:
        _bm25_index = None
        _indexed_chunk_ids = []
        return None, []
        
    _indexed_chunk_ids = [c['id'] for c in chunks]
    tokenized_corpus = [tokenize_corpus(c['text']) for c in chunks]
    
    _bm25_index = BM25Plus(tokenized_corpus)
    return _bm25_index, _indexed_chunk_ids

def bm25_search(query, top_k=20, force_rebuild=False):
    """
    Performs sparse keyword retrieval using BM25Plus.
    Returns: list of dicts [{'chunk_id': id, 'score': bm25_score, 'rank': rank}]
    """
    index, chunk_ids = get_or_build_bm25_index(force_rebuild=force_rebuild)
    if index is None or not chunk_ids:
        return []
        
    query_tokens = tokenize_corpus(query)
    if not query_tokens:
        return []
        
    doc_scores = index.get_scores(query_tokens)
    scored_pairs = list(zip(chunk_ids, doc_scores))
    
    # Sort descending by score
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Extract only documents where score is higher than baseline (i.e. term matched)
    min_score = min(doc_scores) if len(doc_scores) > 0 else 0
    
    results = []
    for rank, (cid, score) in enumerate(scored_pairs[:top_k], start=1):
        if score > min_score or (len(scored_pairs) == 1 and score > 0):
            results.append({
                'chunk_id': cid,
                'score': float(score),
                'rank': rank
            })
            
    return results
