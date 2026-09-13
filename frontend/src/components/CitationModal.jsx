import React, { useState } from 'react';
import { X, FileText, Bookmark, Layers, CheckCircle, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react';

export default function CitationModal({ citation, onClose }) {
  if (!citation) return null;

  const [showParent, setShowParent] = useState(false);

  return (
    <div className="explain-modal-backdrop" onClick={onClose}>
      <div 
        className="explain-modal" 
        style={{ maxWidth: '680px' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="explain-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div style={{ 
              width: '32px', 
              height: '32px', 
              borderRadius: '8px', 
              background: 'rgba(99, 102, 241, 0.2)', 
              color: 'var(--primary)',
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center' 
            }}>
              <FileText size={18} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.05rem', color: '#fff' }}>
                Verified Source Citation [{citation.citation_number || 1}]
              </h2>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-dim)' }}>
                Verifiable grounding from indexed knowledge base
              </div>
            </div>
          </div>
          <button 
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer' }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="explain-body">
          {/* Metadata Badges */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
            <span style={{ 
              background: 'rgba(99, 102, 241, 0.15)', 
              color: '#c7d2fe', 
              border: '1px solid rgba(99, 102, 241, 0.3)',
              padding: '0.25rem 0.65rem', 
              borderRadius: '6px', 
              fontSize: '0.78rem',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}>
              📄 {citation.document_filename}
            </span>

            <span style={{ 
              background: 'rgba(6, 182, 212, 0.15)', 
              color: '#38bdf8', 
              border: '1px solid rgba(6, 182, 212, 0.3)',
              padding: '0.25rem 0.65rem', 
              borderRadius: '6px', 
              fontSize: '0.78rem',
              fontWeight: 600
            }}>
              Page {citation.page_number || 1}
            </span>

            {citation.heading && (
              <span style={{ 
                background: 'rgba(168, 85, 247, 0.15)', 
                color: '#d8b4fe', 
                border: '1px solid rgba(168, 85, 247, 0.3)',
                padding: '0.25rem 0.65rem', 
                borderRadius: '6px', 
                fontSize: '0.78rem',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem'
              }}>
                <Bookmark size={12} />
                {citation.heading}
              </span>
            )}

            <span style={{ 
              background: 'rgba(16, 185, 129, 0.15)', 
              color: '#34d399', 
              border: '1px solid rgba(16, 185, 129, 0.3)',
              padding: '0.25rem 0.65rem', 
              borderRadius: '6px', 
              fontSize: '0.78rem',
              fontWeight: 600
            }}>
              Cross-Encoder Match: {citation.relevance || `${(citation.relevance_score * 100).toFixed(1)}%`}
            </span>
          </div>

          {/* Exact Matching Passage */}
          <div className="explain-section">
            <div className="explain-section-title">
              <CheckCircle size={15} style={{ color: '#10b981' }} />
              Exact Retrieved Passage (Child Chunk #{citation.chunk_id})
            </div>
            <div style={{ 
              background: 'rgba(15, 23, 42, 0.7)', 
              border: '1px solid rgba(99, 102, 241, 0.25)', 
              padding: '1rem', 
              borderRadius: '8px', 
              fontSize: '0.86rem', 
              color: '#f1f5f9',
              lineHeight: '1.6',
              whiteSpace: 'pre-wrap'
            }}>
              {citation.exact_snippet}
            </div>
          </div>

          {/* Parent Context Expansion (Hierarchical Chunking) */}
          {citation.expanded_parent_snippet && (
            <div className="explain-section">
              <div 
                className="explain-section-title"
                style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                onClick={() => setShowParent(!showParent)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Layers size={15} style={{ color: 'var(--primary)' }} />
                  <span>Expanded Parent Section Context (Fed to LLM)</span>
                </div>
                {showParent ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </div>

              {showParent && (
                <div style={{ 
                  background: 'rgba(0, 0, 0, 0.3)', 
                  border: '1px solid var(--border-subtle)', 
                  padding: '1rem', 
                  borderRadius: '8px', 
                  fontSize: '0.82rem', 
                  color: '#94a3b8',
                  lineHeight: '1.6',
                  whiteSpace: 'pre-wrap',
                  animation: 'fadeIn 0.2s ease'
                }}>
                  {citation.expanded_parent_snippet}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
