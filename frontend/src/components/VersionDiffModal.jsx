import React, { useState } from 'react';
import { 
  GitCompare, 
  PlusCircle, 
  MinusCircle, 
  Edit3, 
  CheckCircle2, 
  Search, 
  RefreshCw, 
  FileText, 
  ArrowRight,
  Sparkles
} from 'lucide-react';

export default function VersionDiffModal({ docA, docB, onClose }) {
  const [topicQuery, setTopicQuery] = useState('');
  const [diffData, setDiffData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch diff on initial mount or when requested
  const handleCompare = async (topic = topicQuery) => {
    if (!docA || !docB) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/documents/rag/compare/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id_v1: docA.id,
          doc_id_v2: docB.id,
          topic_query: topic
        })
      });

      if (res.ok) {
        const data = await res.json();
        setDiffData(data);
      } else {
        const err = await res.json().catch(() => ({}));
        setError(err.error || 'Failed to compare document versions');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    handleCompare('');
  }, [docA, docB]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-container diff-modal" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-group">
            <div className="diff-modal-badge">
              <GitCompare size={16} />
              <span>Semantic Document Diff</span>
            </div>
            <h3>Version Comparison & Clause Change Tracking</h3>
            <div className="version-compare-banner">
              <div className="ver-pill ver-a">
                <FileText size={14} />
                <span>{docA?.filename}</span>
                <span className="ver-tag">v{docA?.version_number || 1}</span>
              </div>
              <ArrowRight size={16} className="arrow-sep" />
              <div className="ver-pill ver-b">
                <FileText size={14} />
                <span>{docB?.filename}</span>
                <span className="ver-tag">v{docB?.version_number || 2}</span>
              </div>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Search / Topic Filter Bar */}
        <div className="diff-filter-bar">
          <div className="diff-input-wrapper">
            <Search size={16} className="diff-search-icon" />
            <input 
              type="text"
              className="diff-topic-input"
              placeholder="Filter diff by topic or clause (e.g. 'work experience', 'remote work', 'compensation')..."
              value={topicQuery}
              onChange={e => setTopicQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCompare(topicQuery)}
            />
          </div>
          <button 
            className="diff-compare-btn" 
            onClick={() => handleCompare(topicQuery)}
            disabled={loading}
          >
            {loading ? <RefreshCw size={15} className="spin" /> : <Sparkles size={15} />}
            <span>Compare</span>
          </button>
        </div>

        {/* Content Body */}
        <div className="modal-body diff-modal-body">
          {error && <div className="diff-error-box">{error}</div>}

          {loading && (
            <div className="diff-loading-box">
              <RefreshCw size={24} className="spin" />
              <p>Computing semantic cross-version vector alignment & identifying clause diffs...</p>
            </div>
          )}

          {!loading && diffData && (
            <>
              {/* Metrics Summary Strip */}
              <div className="diff-metrics-strip">
                <div className="diff-metric-pill added">
                  <PlusCircle size={15} />
                  <span>{diffData.metrics?.added_count || 0} Added Clauses</span>
                </div>
                <div className="diff-metric-pill modified">
                  <Edit3 size={15} />
                  <span>{diffData.metrics?.modified_count || 0} Modified Clauses</span>
                </div>
                <div className="diff-metric-pill removed">
                  <MinusCircle size={15} />
                  <span>{diffData.metrics?.removed_count || 0} Removed Clauses</span>
                </div>
                <div className="diff-metric-pill unchanged">
                  <CheckCircle2 size={15} />
                  <span>{diffData.metrics?.unchanged_count || 0} Preserved Provisions</span>
                </div>
              </div>

              {/* Executive Summary Box */}
              {diffData.summary_diff && (
                <div className="diff-executive-summary">
                  <div className="diff-summary-title">
                    <Sparkles size={15} />
                    <span>Executive AI Diff Summary</span>
                  </div>
                  <p>{diffData.summary_diff}</p>
                </div>
              )}

              {/* Clause Diff Sections */}
              <div className="diff-clauses-container">
                {/* 1. Newly Added Clauses */}
                {diffData.added_clauses && diffData.added_clauses.length > 0 && (
                  <div className="diff-section-block">
                    <h4 className="diff-section-title added-title">
                      <PlusCircle size={16} /> Newly Added in v{docB?.version_number || 2}
                    </h4>
                    <div className="diff-clause-list">
                      {diffData.added_clauses.map((item, idx) => (
                        <div key={idx} className="diff-clause-card added-card">
                          <span className="diff-marker plus">+</span>
                          <p>{item.clause}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 2. Modified Clauses */}
                {diffData.modified_clauses && diffData.modified_clauses.length > 0 && (
                  <div className="diff-section-block">
                    <h4 className="diff-section-title modified-title">
                      <Edit3 size={16} /> Modified & Updated Clauses
                    </h4>
                    <div className="diff-clause-list">
                      {diffData.modified_clauses.map((item, idx) => (
                        <div key={idx} className="diff-clause-card modified-card">
                          <div className="modified-subblock old">
                            <span className="ver-badge">v{docA?.version_number || 1}</span>
                            <p>{item.v1_original}</p>
                          </div>
                          <div className="modified-subblock new">
                            <span className="ver-badge">v{docB?.version_number || 2}</span>
                            <p>{item.v2_updated}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 3. Removed Clauses */}
                {diffData.removed_clauses && diffData.removed_clauses.length > 0 && (
                  <div className="diff-section-block">
                    <h4 className="diff-section-title removed-title">
                      <MinusCircle size={16} /> Removed from v{docA?.version_number || 1}
                    </h4>
                    <div className="diff-clause-list">
                      {diffData.removed_clauses.map((item, idx) => (
                        <div key={idx} className="diff-clause-card removed-card">
                          <span className="diff-marker minus">-</span>
                          <p>{item.clause}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {(!diffData.added_clauses?.length && !diffData.modified_clauses?.length && !diffData.removed_clauses?.length) && (
                  <div className="diff-empty-state">
                    <CheckCircle2 size={32} style={{ color: 'var(--success)' }} />
                    <p>No semantic differences detected between these two document sections.</p>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <button className="secondary-btn" onClick={onClose}>Close Inspector</button>
        </div>
      </div>
    </div>
  );
}
