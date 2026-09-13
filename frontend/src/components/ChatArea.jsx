import React, { useRef, useEffect } from 'react';
import { Send, Bot, User, BrainCircuit, Sparkles, RefreshCw, Bookmark, Layers } from 'lucide-react';

export default function ChatArea({ 
  messages, 
  inputValue, 
  setInputValue, 
  onSendMessage, 
  isStreaming, 
  onInspectMessage,
  onOpenCitation
}) {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isStreaming) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  // Helper to render content with clickable inline citation badges [1], [2]
  const renderMessageContent = (text, citations = []) => {
    if (!citations || citations.length === 0) {
      return text;
    }

    // Split text by inline citation tags like [1], [2], [3]
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((part, i) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const citationNum = parseInt(match[1], 10);
        const targetCitation = citations.find(c => c.citation_number === citationNum) || citations[citationNum - 1];
        if (targetCitation) {
          return (
            <button
              key={i}
              onClick={() => onOpenCitation(targetCitation)}
              style={{
                background: 'rgba(99, 102, 241, 0.25)',
                border: '1px solid rgba(99, 102, 241, 0.5)',
                color: '#a5b4fc',
                fontSize: '0.72rem',
                fontWeight: 700,
                padding: '0.1rem 0.4rem',
                borderRadius: '4px',
                cursor: 'pointer',
                margin: '0 0.15rem',
                verticalAlign: 'baseline',
                transition: 'all 0.15s ease'
              }}
              title={`Click to inspect source: ${targetCitation.document_filename} (Page ${targetCitation.page_number || 1})`}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'var(--primary)';
                e.currentTarget.style.color = '#fff';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(99, 102, 241, 0.25)';
                e.currentTarget.style.color = '#a5b4fc';
              }}
            >
              [{citationNum}]
            </button>
          );
        }
      }
      return part;
    });
  };

  return (
    <div className="chat-container">
      {/* Messages Area */}
      <div className="messages-area">
        {messages.length === 0 ? (
          <div style={{ margin: 'auto', textAlign: 'center', maxWidth: '520px' }}>
            <div style={{ 
              width: '56px', 
              height: '56px', 
              borderRadius: '16px', 
              background: 'rgba(99, 102, 241, 0.15)', 
              color: 'var(--primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1.25rem'
            }}>
              <Sparkles size={28} />
            </div>
            <h2 style={{ fontSize: '1.4rem', marginBottom: '0.5rem', color: '#fff' }}>
              RAG Intelligence Assistant
            </h2>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
              Conversational RAG with Multi-Turn Query Rewriting, Parent-Child Hierarchical Chunking, and Deep Verifiable Citations.
            </p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div key={index} className={`message-row ${msg.role}`}>
              <div className={`message-avatar ${msg.role === 'assistant' ? 'avatar-ai' : 'avatar-user'}`}>
                {msg.role === 'assistant' ? <Bot size={18} /> : <User size={18} />}
              </div>

              <div className="message-bubble">
                {/* Header row: Intent & Query Rewriter Notification */}
                {msg.role === 'assistant' && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center', marginBottom: '0.75rem' }}>
                    {msg.intent && (
                      <span className={`intent-badge intent-${msg.intent.toLowerCase()}`} style={{ marginBottom: 0 }}>
                        ● {msg.intent} {msg.confidence ? `(${(msg.confidence * 100).toFixed(0)}%)` : ''}
                      </span>
                    )}

                    {/* Query Rewriter Indicator */}
                    {msg.rewriteInfo && msg.rewriteInfo.was_rewritten && (
                      <span style={{ 
                        fontSize: '0.72rem', 
                        background: 'rgba(168, 85, 247, 0.12)', 
                        border: '1px solid rgba(168, 85, 247, 0.3)', 
                        color: '#d8b4fe',
                        padding: '0.18rem 0.55rem',
                        borderRadius: '9999px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.35rem'
                      }}
                      title={msg.rewriteInfo.reason}
                      >
                        <RefreshCw size={10} />
                        Disambiguated to: "{msg.rewriteInfo.standalone_query.slice(0, 45)}..."
                      </span>
                    )}
                  </div>
                )}

                {/* Message Content with Clickable Footnotes */}
                <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6', fontSize: '0.92rem' }}>
                  {msg.role === 'assistant' 
                    ? renderMessageContent(msg.content, msg.citations) 
                    : msg.content}
                </div>

                {/* Verifiable Citation Chips */}
                {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                  <div className="citations-wrapper">
                    <div className="citation-title">
                      Verified Source Citations (Click to inspect exact passage):
                    </div>
                    <div className="citation-pills">
                      {msg.citations.map((c, cIdx) => (
                        <div 
                          key={cIdx} 
                          className="citation-chip" 
                          onClick={() => onOpenCitation(c)}
                          title={`Click to inspect passage in ${c.document_filename}`}
                        >
                          <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>
                            [{c.citation_number || cIdx + 1}]
                          </span>
                          <span>📄 {c.document_filename} (p. {c.page_number || 1})</span>
                          {c.heading && (
                            <span style={{ color: '#d8b4fe', fontSize: '0.7rem' }}>
                              • {c.heading}
                            </span>
                          )}
                          <span style={{ color: '#34d399', fontSize: '0.7rem' }}>
                            {c.relevance || `${(c.relevance_score * 100).toFixed(0)}%`}
                          </span>
                          {c.is_table && (
                            <span style={{ background: 'rgba(6, 182, 212, 0.2)', color: '#06b6d4', padding: '0 4px', borderRadius: '3px', fontSize: '0.65rem' }}>
                              TABLE
                            </span>
                          )}
                          {c.has_parent_expanded && (
                            <span style={{ color: '#a5b4fc', fontSize: '0.65rem' }} title="Parent context expanded">
                              <Layers size={10} />
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Explainability / MLOps Button */}
                {msg.role === 'assistant' && (msg.explainabilityData || msg.metrics) && (
                  <div>
                    <button 
                      className="btn-explain"
                      onClick={() => onInspectMessage(msg)}
                    >
                      <BrainCircuit size={13} />
                      <span>Inspect AI Thought Process & MLOps</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="input-container">
        <form onSubmit={handleSubmit} className="input-box">
          <input
            type="text"
            className="chat-input"
            placeholder={isStreaming ? "Streaming response token-by-token..." : "Ask a question, or test follow-up questions with conversational memory..."}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={isStreaming}
          />
          <button 
            type="submit" 
            className="btn-send"
            disabled={isStreaming || !inputValue.trim()}
          >
            <Send size={16} />
          </button>
        </form>
      </div>
    </div>
  );
}
