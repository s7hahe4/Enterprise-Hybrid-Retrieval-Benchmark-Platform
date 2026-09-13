import os
import re

AMBIGUOUS_PATTERNS = [
    r'\b(he|she|it|they|his|her|their|that|this|these|those|its|him|them)\b',
    r'^\s*(what\s*about|how\s*about|and\s*what|and\s*how|why\s*is\s*that|where\s*did|who\s*is\s*he|tell\s*me\s*more)\b',
    r'^\s*(what\s*else|more\s*details|compare|what\s*about\s*\d{4})\b',
]

def extract_key_entities(text):
    """Extracts prominent capitalized proper nouns and domain entities from text."""
    words = re.findall(r'\b[A-Z][a-zA-Z0-9_\-]+\b', text)
    # Filter common sentence-starters
    stop_starts = {'The', 'This', 'That', 'These', 'Those', 'What', 'How', 'Where', 'When', 'Why', 'Who', 'Based', 'According', 'Yes', 'No'}
    return [w for w in words if w not in stop_starts]

def rewrite_query(current_query, conversation_history=None):
    """
    Analyzes conversation context and reformulates ambiguous follow-up questions
    into unambiguous standalone queries for vector and lexical retrieval.
    
    Returns:
        dict: {
            'standalone_query': str,
            'was_rewritten': bool,
            'reason': str
        }
    """
    if not conversation_history or len(conversation_history) == 0:
        return {
            'standalone_query': current_query,
            'was_rewritten': False,
            'reason': 'First message in session; standalone by default.'
        }

    q_lower = current_query.lower().strip()
    
    # Check if query exhibits coreference or conversational ambiguity
    is_ambiguous = any(re.search(p, q_lower) for p in AMBIGUOUS_PATTERNS) or len(current_query.split()) <= 4
    
    if not is_ambiguous:
        return {
            'standalone_query': current_query,
            'was_rewritten': False,
            'reason': 'Query contains explicit standalone domain terms.'
        }
        
    # If OpenAI API Key is available, use fast LLM reformulation
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            
            recent_turns = conversation_history[-4:]
            formatted_history = "\n".join([
                f"{m['role'].capitalize()}: {m['content']}" 
                for m in recent_turns
            ])
            
            prompt = (
                f"Given the following conversation history and a follow-up question, "
                f"rephrase the follow-up question into a single, standalone search query "
                f"that can be understood completely without the prior conversation.\n"
                f"Do NOT answer the question. Only output the reformulated question.\n\n"
                f"Conversation History:\n{formatted_history}\n\n"
                f"Follow-up Question: {current_query}\n\n"
                f"Standalone Query:"
            )
            
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            
            rewritten = resp.choices[0].message.content.strip().strip('"')
            if rewritten and rewritten.lower() != current_query.lower():
                return {
                    'standalone_query': rewritten,
                    'was_rewritten': True,
                    'reason': 'LLM context disambiguation resolved conversational references.'
                }
        except Exception:
            pass

    # High-Precision Heuristic Rewriting Fallback (Offline / Local Mode)
    recent_user_queries = [m['content'] for m in conversation_history if m['role'] == 'user']
    recent_assistant_replies = [m['content'] for m in conversation_history if m['role'] == 'assistant']
    
    last_user_query = recent_user_queries[-1] if recent_user_queries else ""
    last_reply = recent_assistant_replies[-1] if recent_assistant_replies else ""
    
    entities = extract_key_entities(last_user_query) + extract_key_entities(last_reply[:300])
    main_subject = entities[0] if entities else "the candidate / document"
    
    # Reformulate patterns like "What about deep learning?" -> "What about deep learning regarding {subject}?"
    if re.match(r'^\s*(what|how)\s*about\s+(.+)', q_lower):
        topic = re.sub(r'^\s*(what|how)\s*about\s+', '', current_query, flags=re.IGNORECASE).rstrip('?')
        standalone = f"What experience or information is recorded about {topic} regarding {main_subject}?"
        return {
            'standalone_query': standalone,
            'was_rewritten': True,
            'reason': f"Resolved follow-up context around primary entity '{main_subject}'."
        }
        
    if re.search(r'\b(he|his|him)\b', q_lower):
        standalone = re.sub(r'\b(he|his|him)\b', main_subject, current_query, flags=re.IGNORECASE)
        return {
            'standalone_query': standalone,
            'was_rewritten': True,
            'reason': f"Resolved third-person pronoun to entity '{main_subject}'."
        }
        
    standalone = f"{main_subject} {current_query}"
    return {
        'standalone_query': standalone,
        'was_rewritten': True,
        'reason': f"Linked conversational context to active subject '{main_subject}'."
    }
