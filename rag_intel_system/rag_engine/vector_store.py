import faiss
import numpy as np
import os
import hashlib
from django.conf import settings

_model = None
VECTOR_DIMENSION = 384  # 'all-MiniLM-L6-v2' outputs 384-dimensional vectors
INDEX_FILE = str(settings.BASE_DIR / 'faiss_index.bin')

def get_embedding_model():
    """Lazily load the SentenceTransformer model with strict thread limits for 512MB RAM constraints."""
    global _model
    if _model is None:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        try:
            import torch
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer ({e}). Using deterministic fallback.")
            _model = "FALLBACK"
    return _model

def _deterministic_fallback_embed(texts):
    """Fallback 384-d normalized embeddings if PyTorch cannot allocate memory on micro free-tier instances."""
    vectors = []
    for text in texts:
        # Create a stable 384-d pseudo-semantic vector from token n-grams
        v = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = h % VECTOR_DIMENSION
            v[idx] += 1.0 / (1.0 + 0.1 * i)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        vectors.append(v)
    return np.array(vectors, dtype=np.float32)

def get_faiss_index():
    """Reads existing FAISS index from disk or initializes a new IndexIDMap."""
    if os.path.exists(INDEX_FILE):
        try:
            return faiss.read_index(INDEX_FILE)
        except Exception:
            pass
    # Create a new index that links FAISS vector positions directly to Django Chunk IDs
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    return faiss.IndexIDMap(index)

def add_chunks_to_vector_store(django_ids, text_chunks):
    """Encodes text chunks and indexes them with their corresponding Django Chunk IDs."""
    model = get_embedding_model()
    index = get_faiss_index()
    
    # 1. Turn text into embeddings (vectors)
    if model == "FALLBACK":
        embeddings = _deterministic_fallback_embed(text_chunks)
    else:
        try:
            import torch
            with torch.no_grad():
                embeddings = model.encode(
                    text_chunks, 
                    batch_size=4, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
        except Exception as e:
            print(f"Embedding batch failed ({e}), using deterministic fallback embeddings.")
            embeddings = _deterministic_fallback_embed(text_chunks)
    
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
    if model == "FALLBACK":
        query_vector = _deterministic_fallback_embed([query])
    else:
        try:
            import torch
            with torch.no_grad():
                query_vector = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        except Exception:
            query_vector = _deterministic_fallback_embed([query])
    
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