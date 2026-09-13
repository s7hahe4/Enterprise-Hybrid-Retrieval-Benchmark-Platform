import faiss
import numpy as np
import os

_model = None
VECTOR_DIMENSION = 384  # 'all-MiniLM-L6-v2' outputs 384-dimensional vectors
INDEX_FILE = 'faiss_index.bin'

def get_embedding_model():
    """Lazily load the SentenceTransformer model on first usage so server startup is instantaneous."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_faiss_index():
    """Reads existing FAISS index from disk or initializes a new IndexIDMap."""
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    else:
        # Create a new index that links FAISS vector positions directly to Django Chunk IDs
        index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        return faiss.IndexIDMap(index)

def add_chunks_to_vector_store(django_ids, text_chunks):
    """Encodes text chunks and indexes them with their corresponding Django Chunk IDs."""
    model = get_embedding_model()
    index = get_faiss_index()
    
    # 1. Turn text into embeddings (vectors)
    embeddings = model.encode(text_chunks)
    
    # 2. Add to FAISS index with corresponding Django database IDs
    index.add_with_ids(
        np.array(embeddings, dtype=np.float32), 
        np.array(django_ids, dtype=np.int64)
    )
    
    # 3. Persist the index to disk
    faiss.write_index(index, INDEX_FILE)

def dense_vector_search(query, top_k=20):
    """
    Performs dense vector similarity search over the FAISS index.
    Returns: list of dicts [{'chunk_id': id, 'distance': float(dist), 'rank': rank}]
    """
    if not os.path.exists(INDEX_FILE):
        return []
        
    index = get_faiss_index()
    if index.ntotal == 0:
        return []
        
    model = get_embedding_model()
    query_vector = model.encode([query])
    
    k = min(top_k, index.ntotal)
    distances, indices = index.search(
        np.array(query_vector, dtype=np.float32),
        k
    )
    
    results = []
    for rank, (chunk_id, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if chunk_id != -1:  # -1 indicates unassigned slot in FAISS
            results.append({
                'chunk_id': int(chunk_id),
                'distance': float(dist),
                'similarity': float(1 / (1 + dist)),  # normalized similarity [0, 1]
                'rank': rank
            })
            
    return results