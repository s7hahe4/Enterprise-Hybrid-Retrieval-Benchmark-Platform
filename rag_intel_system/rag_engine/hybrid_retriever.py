from documents.models import Chunk
from .vector_store import dense_vector_search
from .bm25_search import bm25_search

RRF_K = 60  # Standard constant used in Reciprocal Rank Fusion

ROLE_HIERARCHY = {
    'ADMIN': {'PUBLIC', 'ENGINEERING', 'HR', 'FINANCE', 'ADMIN'},
    'ENGINEERING': {'PUBLIC', 'ENGINEERING'},
    'HR': {'PUBLIC', 'HR'},
    'FINANCE': {'PUBLIC', 'FINANCE'},
    'PUBLIC': {'PUBLIC'}
}

class CandidateList(list):
    """
    Subclasses built-in list to preserve exact sequence semantics, JSON serializability,
    and slicing while attaching RBAC security telemetry (blocked_count, user_role).
    """
    def __init__(self, items=None, blocked_count=0, user_role='PUBLIC'):
        super().__init__(items or [])
        self.blocked_count = blocked_count
        self.user_role = user_role

def get_allowed_roles_for_user(user_role: str):
    """Computes access permissions for a given user security tier."""
    role_clean = (user_role or 'PUBLIC').strip().upper()
    return ROLE_HIERARCHY.get(role_clean, {'PUBLIC'})

def hybrid_retrieve_candidates(query, top_k=20, user_role='PUBLIC'):
    """
    Executes hybrid retrieval combining dense FAISS search and sparse BM25 search
    using Reciprocal Rank Fusion (RRF) with enterprise RBAC security filtering.
    """
    dense_results = dense_vector_search(query, top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)
    
    rrf_scores = {}
    metadata_map = {}
    
    # Process dense results
    for item in dense_results:
        cid = item['chunk_id']
        rank = item['rank']
        score = 1.0 / (RRF_K + rank)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
        
        metadata_map[cid] = {
            'dense_rank': rank,
            'dense_distance': round(item['distance'], 4),
            'dense_similarity': round(item['similarity'], 4),
            'bm25_rank': None,
            'bm25_score': 0.0
        }
        
    # Process BM25 results
    for item in bm25_results:
        cid = item['chunk_id']
        rank = item['rank']
        score = 1.0 / (RRF_K + rank)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
        
        if cid in metadata_map:
            metadata_map[cid]['bm25_rank'] = rank
            metadata_map[cid]['bm25_score'] = round(item['score'], 4)
        else:
            metadata_map[cid] = {
                'dense_rank': None,
                'dense_distance': None,
                'dense_similarity': 0.0,
                'bm25_rank': rank,
                'bm25_score': round(item['score'], 4)
            }
            
    if not rrf_scores:
        return CandidateList([], blocked_count=0, user_role=user_role)
        
    # Sort candidate IDs by fused RRF score
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    candidate_ids = [cid for cid, _ in sorted_candidates]
    allowed_roles = get_allowed_roles_for_user(user_role)
    
    # Bulk fetch chunk records from SQLite
    chunks_query = Chunk.objects.filter(id__in=candidate_ids).select_related('document')
    chunk_db_map = {c.id: c for c in chunks_query}
    
    candidates = []
    blocked_count = 0
    
    for rrf_rank, (cid, rrf_score) in enumerate(sorted_candidates, start=1):
        chunk = chunk_db_map.get(cid)
        if not chunk:
            continue
            
        doc_role = getattr(chunk.document, 'access_role', 'PUBLIC') or 'PUBLIC'
        if doc_role.upper() not in allowed_roles:
            blocked_count += 1
            continue
            
        meta = metadata_map[cid]
        candidates.append({
            'chunk_id': chunk.id,
            'chunk_index': chunk.chunk_index,
            'document_id': chunk.document.id,
            'document_filename': chunk.document.filename,
            'document_role': doc_role,
            'text': chunk.text,
            'is_table': chunk.is_table,
            'metadata': chunk.metadata,
            'rrf_score': round(rrf_score, 6),
            'initial_rrf_rank': rrf_rank,
            'dense_rank': meta['dense_rank'],
            'dense_distance': meta['dense_distance'],
            'dense_similarity': meta['dense_similarity'],
            'bm25_rank': meta['bm25_rank'],
            'bm25_score': meta['bm25_score']
        })
        
    return CandidateList(candidates, blocked_count=blocked_count, user_role=user_role)


