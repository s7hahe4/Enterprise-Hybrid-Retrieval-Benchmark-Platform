import re
from .vector_store import dense_vector_search

GREETING_PATTERNS = [
    r'^\s*(hi|hello|hey|greetings|howdy|good\s*(morning|afternoon|evening))\b',
    r'^\s*(how\s*are\s*you|who\s*are\s*you|what\s*are\s*you|what\s*can\s*you\s*do|help)\b',
    r'^\s*(thanks|thank\s*you|bye|goodbye)\b'
]

CHITCHAT_RESPONSES = {
    'greeting': "Hello! I am your RAG Intelligence Assistant. I am equipped with neural hybrid retrieval (FAISS + BM25) and Cross-Encoder re-ranking to accurately answer questions about your uploaded documents. What would you like to explore?",
    'identity': "I am an enterprise RAG assistant with agentic intent guardrails, hybrid search, and real-time explainability. You can ask me questions about any of your ingested PDF documents.",
    'gratitude': "You're very welcome! Feel free to ask more questions about your indexed documents.",
    'farewell': "Goodbye! Have a productive day."
}

def classify_intent(query):
    """
    Agentic Intent Router & Guardrail Layer:
    Determines whether a user query is:
    1. 'CHITCHAT': General greeting or conversational query (handled instantly).
    2. 'IN_DOMAIN_RAG': Relates to context stored in indexed documents.
    3. 'OUT_OF_DOMAIN': Irrelevant or out-of-distribution question (safely rejected).
    
    Returns dict:
      {
         'intent': 'IN_DOMAIN_RAG' | 'CHITCHAT' | 'OUT_OF_DOMAIN',
         'confidence': float,
         'rationale': str,
         'precomputed_response': Optional[str]
      }
    """
    cleaned_query = query.strip().lower()
    
    # Check 1: Greetings & Chitchat
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, cleaned_query):
            if any(w in cleaned_query for w in ['who', 'what can', 'what are']):
                resp = CHITCHAT_RESPONSES['identity']
            elif any(w in cleaned_query for w in ['thank', 'thanks']):
                resp = CHITCHAT_RESPONSES['gratitude']
            elif any(w in cleaned_query for w in ['bye', 'goodbye']):
                resp = CHITCHAT_RESPONSES['farewell']
            else:
                resp = CHITCHAT_RESPONSES['greeting']
                
            return {
                'intent': 'CHITCHAT',
                'confidence': 0.98,
                'rationale': 'Query matches conversational intent pattern. Bypassing database retrieval to save latency and prevent hallucination.',
                'precomputed_response': resp
            }

    # Check 2: Hybrid Semantic & Keyword Domain Boundary Check
    top_candidates = dense_vector_search(query, top_k=3)
    from .bm25_search import bm25_search
    top_sparse = bm25_search(query, top_k=3)
    
    if not top_candidates and not top_sparse:
        return {
            'intent': 'OUT_OF_DOMAIN',
            'confidence': 0.95,
            'rationale': 'No documents have been indexed into the vector store yet.',
            'precomputed_response': "No documents are currently indexed in the knowledge base. Please upload a PDF document first so I can retrieve relevant context."
        }
        
    best_sim = top_candidates[0]['similarity'] if top_candidates else 0.0
    has_keyword_match = len(top_sparse) > 0 and top_sparse[0].get('score', 0) > 0.2
    
    # OOD Threshold: calibrated for general document exploration
    OOD_THRESHOLD_SIMILARITY = 0.28
    
    if best_sim < OOD_THRESHOLD_SIMILARITY and not has_keyword_match:
        return {
            'intent': 'OUT_OF_DOMAIN',
            'confidence': round(1.0 - best_sim, 3),
            'rationale': f"Query semantic similarity ({best_sim:.3f}) falls below domain relevance threshold ({OOD_THRESHOLD_SIMILARITY}) with no keyword matches. Guardrail triggered to prevent out-of-distribution hallucination.",
            'precomputed_response': "I am specifically scoped to answer questions grounded in your uploaded documents. Your query appears to be outside the domain of the indexed knowledge base. Please ask a question related to your uploaded PDFs."
        }
        
    return {
        'intent': 'IN_DOMAIN_RAG',
        'confidence': round(min(0.99, max(best_sim + 0.30, 0.85)), 3),
        'rationale': f"Query passed domain guardrails (semantic match: {best_sim:.3f}, keyword hit: {has_keyword_match}). Routing to Hybrid Retrieval + Cross-Encoder pipeline.",
        'precomputed_response': None
    }
