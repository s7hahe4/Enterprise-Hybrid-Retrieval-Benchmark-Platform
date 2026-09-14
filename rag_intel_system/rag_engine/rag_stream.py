import os
import json
import time
from documents.models import Chunk, QueryLog, QueryAuditLog
from .router import classify_intent
from .hybrid_retriever import hybrid_retrieve_candidates
from .reranker import rerank_chunks
from .evaluator import evaluate_ragas_metrics
from .rewriter import rewrite_query

def format_sse(event_type, payload):
    """Formats payload as Server-Sent Event (SSE) JSON."""
    return f"data: {json.dumps({'type': event_type, 'payload': payload})}\n\n"

def _stream_rag_pipeline_inner(query, conversation_history=None, user_role='PUBLIC'):
    """
    Enterprise RAG Streaming Pipeline with Multi-Turn Conversational Memory & RBAC:
    0. Multi-Turn Query Rewriting (entity resolution & pronoun disambiguation)
    1. Agentic Intent Classification & Semantic Guardrails
    2. Hybrid Retrieval (FAISS Dense + BM25 Sparse with RRF & RBAC Filtering)
    3. Cross-Encoder Passage Re-ranking
    4. Hierarchical Parent-Child Context Expansion
    5. Token-by-token synthesis stream (OpenAI with Grounded Local Fallback)
    6. Real-time RAGAS Evaluation Metrics & MLOps Audit Logging
    """
    start_time = time.time()
    latency_breakdown = {}
    
    # -------------------------------------------------------------
    # Stage 0: Conversational Memory & Query Rewriting
    # -------------------------------------------------------------
    t0 = time.time()
    rewrite_result = rewrite_query(query, conversation_history)
    t1 = time.time()
    latency_breakdown['rewrite_ms'] = round((t1 - t0) * 1000, 1)
    
    active_query = rewrite_result['standalone_query']
    
    yield format_sse('rewrite', {
        'original_query': query,
        'standalone_query': active_query,
        'was_rewritten': rewrite_result['was_rewritten'],
        'reason': rewrite_result['reason'],
        'latency_ms': latency_breakdown['rewrite_ms']
    })
    
    # -------------------------------------------------------------
    # Stage 1: Intent Routing & Guardrails
    # -------------------------------------------------------------
    t0 = time.time()
    routing_result = classify_intent(active_query)
    t1 = time.time()
    latency_breakdown['intent_ms'] = round((t1 - t0) * 1000, 1)
    
    yield format_sse('intent', {
        'intent': routing_result['intent'],
        'confidence': routing_result['confidence'],
        'rationale': routing_result['rationale'],
        'latency_ms': latency_breakdown['intent_ms']
    })
    
    # If Chitchat or Out of Domain: stream immediate precomputed response and finish
    if routing_result['intent'] != 'IN_DOMAIN_RAG':
        full_answer = routing_result['precomputed_response'] or ""
        words = full_answer.split(' ')
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield format_sse('token', {'delta': token})
            time.sleep(0.02)
            
        total_latency = round((time.time() - start_time) * 1000, 1)
        latency_breakdown['total_ms'] = total_latency
        
        # Log to DB
        QueryLog.objects.create(question=query, answer=full_answer)
        QueryAuditLog.objects.create(
            original_query=query,
            question=active_query,
            answer=full_answer,
            intent_classified=routing_result['intent'],
            confidence=routing_result['confidence'],
            was_rewritten=rewrite_result['was_rewritten'],
            rewrite_reason=rewrite_result['reason'],
            retrieved_chunks_count=0,
            retrieval_metadata={'action': 'bypassed_retrieval'},
            latency_breakdown_ms=latency_breakdown
        )
        
        yield format_sse('metrics', {
            'latency_breakdown': latency_breakdown,
            'ragas': {'faithfulness': 1.0, 'relevancy': 1.0}
        })
        yield format_sse('done', {'status': 'completed'})
        return

    # -------------------------------------------------------------
    # Stage 2: Hybrid Retrieval (FAISS + BM25 + RRF) with RBAC Guardrail
    # -------------------------------------------------------------
    t0 = time.time()
    candidates = hybrid_retrieve_candidates(active_query, top_k=20, user_role=user_role)
    t1 = time.time()
    latency_breakdown['retrieval_ms'] = round((t1 - t0) * 1000, 1)
    
    blocked_count = getattr(candidates, 'blocked_count', 0)

    yield format_sse('security', {
        'user_role': user_role,
        'blocked_chunks_count': blocked_count,
        'authorized_count': len(candidates)
    })
    
    # If no authorized candidates remain because of security classification
    if not candidates and blocked_count > 0:
        restricted_msg = (
            f"🔒 **Access Restricted:** Relevant information was identified in secured documents, "
            f"but your current active role (`{user_role.upper()}`) is not authorized to access them. "
            f"Please switch to an authorized role ({blocked_count} restricted passage(s) filtered)."
        )
        for word in restricted_msg.split(' '):
            yield format_sse('token', {'delta': word + ' '})
            time.sleep(0.015)

        total_latency = round((time.time() - start_time) * 1000, 1)
        latency_breakdown['total_ms'] = total_latency

        QueryAuditLog.objects.create(
            original_query=query,
            question=active_query,
            answer=restricted_msg,
            intent_classified=routing_result['intent'],
            confidence=routing_result['confidence'],
            user_role=user_role,
            blocked_chunks_count=blocked_count,
            was_rewritten=rewrite_result['was_rewritten'],
            rewrite_reason=rewrite_result['reason'],
            retrieved_chunks_count=0,
            retrieval_metadata={'action': 'rbac_blocked', 'blocked_chunks': blocked_count},
            latency_breakdown_ms=latency_breakdown,
            ragas_faithfulness=1.0,
            ragas_relevancy=1.0
        )
        yield format_sse('metrics', {
            'latency_breakdown': latency_breakdown,
            'ragas': {'faithfulness': 1.0, 'relevancy': 1.0},
            'citations': [],
            'security': {'user_role': user_role, 'blocked_chunks_count': blocked_count}
        })
        yield format_sse('done', {'status': 'completed'})
        return

    yield format_sse('retrieval', {
        'candidate_count': len(candidates),
        'blocked_count': blocked_count,
        'latency_ms': latency_breakdown['retrieval_ms'],
        'candidates': [
            {
                'chunk_id': c['chunk_id'],
                'document_filename': c['document_filename'],
                'document_role': c.get('document_role', 'PUBLIC'),
                'dense_rank': c['dense_rank'],
                'bm25_rank': c['bm25_rank'],
                'dense_similarity': c['dense_similarity'],
                'bm25_score': c['bm25_score'],
                'rrf_score': c['rrf_score'],
                'initial_rrf_rank': c['initial_rrf_rank'],
                'is_table': c['is_table'],
                'snippet': c['text'][:140] + "..." if len(c['text']) > 140 else c['text']
            }
            for c in candidates[:10]
        ]
    })
    
    # -------------------------------------------------------------
    # Stage 3: Cross-Encoder Re-ranking
    # -------------------------------------------------------------
    t0 = time.time()
    top_reranked, full_reranked = rerank_chunks(active_query, candidates, top_k=4)
    t1 = time.time()
    latency_breakdown['rerank_ms'] = round((t1 - t0) * 1000, 1)
    
    # -------------------------------------------------------------
    # Stage 4: Hierarchical Parent-Child Context Expansion
    # -------------------------------------------------------------
    enriched_citations = []

    llm_context_passages = []
    
    # Fetch full DB Chunk objects to resolve parent chunks and metadata
    top_chunk_ids = [c['chunk_id'] for c in top_reranked]
    db_chunks = {
        c.id: c 
        for c in Chunk.objects.filter(id__in=top_chunk_ids).select_related('parent_chunk', 'document')
    }
    
    for citation_idx, c in enumerate(top_reranked, start=1):
        chunk_obj = db_chunks.get(c['chunk_id'])
        heading = (chunk_obj.heading if chunk_obj else None) or c.get('metadata', {}).get('heading', 'General')
        page_num = (chunk_obj.page_number if chunk_obj else None) or c.get('metadata', {}).get('page', 1)
        
        # Parent-Child Expansion:
        # If chunk is a small child chunk, supply the full unbroken Parent Chunk to the LLM prompt!
        if chunk_obj and chunk_obj.parent_chunk:
            expanded_text = chunk_obj.parent_chunk.text
            has_parent = True
            parent_id = chunk_obj.parent_chunk.id
        else:
            expanded_text = c['text']
            has_parent = False
            parent_id = None
            
        llm_context_passages.append(
            f"[{citation_idx}] [Source: {c['document_filename']}, Page: {page_num}, Section: {heading}]\n{expanded_text}"
        )
        
        enriched_citations.append({
            'citation_number': citation_idx,
            'chunk_id': c['chunk_id'],
            'parent_chunk_id': parent_id,
            'has_parent_expanded': has_parent,
            'document_filename': c['document_filename'],
            'heading': heading,
            'page_number': page_num,
            'is_table': c['is_table'],
            'relevance_score': c['relevance_score'],
            'relevance': f"{c['relevance_score'] * 100:.1f}%",
            'exact_snippet': c['text'],
            'expanded_parent_snippet': expanded_text if has_parent else None
        })
        
    yield format_sse('rerank', {
        'top_count': len(top_reranked),
        'latency_ms': latency_breakdown['rerank_ms'],
        'top_chunks': [
            {
                'chunk_id': c['chunk_id'],
                'document_filename': c['document_filename'],
                'chunk_index': c['chunk_index'],
                'is_table': c['is_table'],
                'relevance_score': c['relevance_score'],
                'initial_rank': c['initial_rrf_rank'],
                'final_rank': c['final_rank'],
                'rank_delta': c['rank_delta'],
                'text': c['text']
            }
            for c in top_reranked
        ]
    })
    
    # -------------------------------------------------------------
    # Stage 5: Token Generation & Streaming (OpenAI or Grounded Synthesis)
    # -------------------------------------------------------------
    t_gen_start = time.time()
    first_token_time = None
    accumulated_answer = []
    
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    stream_successful = False
    
    if openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key, timeout=12.0)
            
            context_text = "\n\n---\n\n".join(llm_context_passages)
            
            system_prompt = (
                "You are an expert enterprise AI research assistant. You are given factual context chunks "
                "retrieved using neural hybrid search and cross-encoder re-ranking with parent-child expansion. "
                "Answer the user's question accurately, concisely, and strictly grounded in the provided context. "
                "CRITICAL: Insert inline citation brackets like [1], [2] immediately following factual claims "
                "to point to the specific numbered sources provided."
            )
            
            # Format multi-turn conversation messages
            messages = [{"role": "system", "content": system_prompt}]
            if conversation_history:
                for h in conversation_history[-4:]:
                    messages.append({"role": h['role'], "content": h['content']})
                    
            messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {active_query}"})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    if first_token_time is None:
                        first_token_time = time.time()
                    accumulated_answer.append(token)
                    yield format_sse('token', {'delta': token})
                    
            stream_successful = True
        except Exception:
            stream_successful = False
            
    if not stream_successful:
        # High-Precision Grounded Local Synthesis Fallback
        if not enriched_citations:
            fallback_text = (
                "I searched the indexed knowledge base, but found no matching passages for your query under your current access role. "
                "Please verify that the relevant document has been uploaded to this security tier, or try rephrasing your search keywords."
            )
        else:
            lead_cite = enriched_citations[0]
            source_name = lead_cite.get('document_filename', 'Indexed Document')
            heading_name = lead_cite.get('heading', 'General')
            page_info = f"Page {lead_cite.get('page_number', 1)}"
            rel_score = f"{lead_cite.get('relevance_score', 0.85)*100:.1f}%"
            
            fallback_text = (
                f"Based on **{source_name}** ({page_info}, Section: *{heading_name}*), "
                f"retrieved with **{rel_score}** semantic confidence [1]:\n\n"
            )
            
            # Extract main sentences from top chunks
            body_text = lead_cite.get('exact_snippet', '').replace('\n', ' ').strip()
            fallback_text += f"{body_text}\n\n"
            
            if len(enriched_citations) > 1:
                second_cite = enriched_citations[1]
                sec_body = second_cite.get('exact_snippet', '').replace('\n', ' ').strip()
                fallback_text += f"**Additional Context [{second_cite.get('citation_number', 2)}]:** {sec_body[:280]}...\n\n"
                
            if any(c.get('is_table') for c in enriched_citations):
                fallback_text += "*(Includes verified tabular data extracted during ingestion)*\n"
            
        words = fallback_text.split(' ')
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            if first_token_time is None:
                first_token_time = time.time()
            accumulated_answer.append(token)
            yield format_sse('token', {'delta': token})
            time.sleep(0.012)
            
    t_gen_end = time.time()
    full_answer = "".join(accumulated_answer)
    
    latency_breakdown['ttft_ms'] = round(((first_token_time or t_gen_end) - t_gen_start) * 1000, 1)
    latency_breakdown['generation_ms'] = round((t_gen_end - t_gen_start) * 1000, 1)
    latency_breakdown['total_ms'] = round((t_gen_end - start_time) * 1000, 1)
    
    # -------------------------------------------------------------
    # Stage 6: Real-time RAGAS Evaluation & MLOps Audit Logging
    # -------------------------------------------------------------
    t0 = time.time()
    ragas_scores = evaluate_ragas_metrics(active_query, full_answer, top_reranked)
    t1 = time.time()
    latency_breakdown['eval_ms'] = round((t1 - t0) * 1000, 1)
    
    yield format_sse('metrics', {
        'latency_breakdown': latency_breakdown,
        'ragas': ragas_scores,
        'citations': enriched_citations
    })
    
    yield format_sse('done', {'status': 'completed'})
    
    # Persist to Audit DB
    try:
        QueryLog.objects.create(question=query, answer=full_answer)
        QueryAuditLog.objects.create(
            original_query=query,
            question=active_query,
            answer=full_answer,
            intent_classified=routing_result['intent'],
            confidence=routing_result['confidence'],
            was_rewritten=rewrite_result['was_rewritten'],
            rewrite_reason=rewrite_result['reason'],
            retrieved_chunks_count=len(top_reranked),
            user_role=user_role,
            blocked_chunks_count=blocked_count,
            retrieval_metadata={
                'top_chunks': [
                    {
                        'id': c['chunk_id'],
                        'doc': c['document_filename'],
                        'relevance': c['relevance_score'],
                        'heading': c.get('heading'),
                        'page': c.get('page_number'),
                        'dense_sim': c.get('dense_similarity'),
                        'bm25_score': c.get('bm25_score'),
                        'rrf_rank': c.get('initial_rrf_rank'),
                        'final_rank': c.get('final_rank')
                    }
                    for c in enriched_citations
                ]
            },
            latency_breakdown_ms=latency_breakdown,
            ragas_faithfulness=ragas_scores['faithfulness'],
            ragas_relevancy=ragas_scores['relevancy']
        )
    except Exception as db_err:
        print(f"Warning: Audit log persistence encountered error: {db_err}")

def stream_rag_pipeline(query, conversation_history=None, user_role='PUBLIC'):
    """
    Resilient SSE generator wrapper.
    Guarantees that unhandled errors are surfaced as user-facing error tokens
    and cleanly finalized with a 'done' event, preventing frontend hangs.
    """
    try:
        yield from _stream_rag_pipeline_inner(query, conversation_history, user_role)
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_msg = f"⚠️ An unexpected error occurred while processing your request: {str(e)}"
        for word in err_msg.split(' '):
            yield format_sse('token', {'delta': word + ' '})
            time.sleep(0.015)
        yield format_sse('done', {'status': 'error', 'error': str(e)})

