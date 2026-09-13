import numpy as np
import re
from .vector_store import get_embedding_model

def cosine_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def evaluate_ragas_metrics(question, answer, context_chunks):
    """
    Computes real-time RAGAS-aligned evaluation metrics:
    1. Faithfulness: Degree to which the answer is grounded in the retrieved context chunks.
    2. Answer Relevancy: Semantic alignment between the user's question and the generated answer.
    
    Returns:
        {'faithfulness': float [0.0 - 1.0], 'relevancy': float [0.0 - 1.0]}
    """
    if not answer or not context_chunks:
        return {'faithfulness': 0.0, 'relevancy': 0.0}
        
    model = get_embedding_model()
    
    # 1. Answer Relevancy: Semantic cosine similarity between Question and Answer embeddings
    q_vec = model.encode([question])[0]
    a_vec = model.encode([answer])[0]
    relevancy = round(max(0.0, min(1.0, (cosine_similarity(q_vec, a_vec) + 1.0) / 2.0)), 3)
    
    # 2. Faithfulness: Groundedness check of answer key terms against context chunks
    combined_context = " ".join([c['text'] for c in context_chunks]).lower()
    answer_words = re.findall(r'\b[a-zA-Z0-9_\-]{4,}\b', answer.lower())
    
    if not answer_words:
        faithfulness = 0.9
    else:
        # Check percentage of significant terms directly present in context
        grounded_terms = [w for w in answer_words if w in combined_context]
        grounding_ratio = len(grounded_terms) / len(answer_words)
        
        # In addition, check vector projection of answer against context
        c_vec = model.encode([combined_context[:2000]])[0]
        context_sim = max(0.0, min(1.0, (cosine_similarity(a_vec, c_vec) + 1.0) / 2.0))
        
        faithfulness = round(0.6 * grounding_ratio + 0.4 * context_sim, 3)
        faithfulness = min(0.99, max(0.1, faithfulness))
        
    return {
        'faithfulness': faithfulness,
        'relevancy': relevancy
    }
