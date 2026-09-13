import React from 'react';
import { X, ShieldAlert, ShieldCheck, Cpu, Database, Gauge, Clock, ArrowUp, ArrowDown, Minus } from 'lucide-react';

export default function ExplainabilityModal({ data, onClose }) {
  if (!data) return null;

  const { intent, retrieval, rerank, metrics, query, answer, security, userRole } = data;
  const ragas = metrics?.ragas || {};
  const latency = metrics?.latency_breakdown || {};
  const blockedCount = security?.blocked_chunks_count ?? retrieval?.blocked_count ?? 0;
  const activeSecurityRole = userRole || security?.user_role || 'PUBLIC';

  return (
    <div className="explain-modal-backdrop" onClick={onClose}>
      <div className="explain-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="explain-header">
          <div>
            <h2 style={{ fontSize: '1.2rem', color: '#fff' }}>
              🧠 MLOps & Explainability Inspector
            </h2>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Deep trace of agentic decisions, hybrid retrieval, RBAC authorization, cross-encoder scores & RAGAS metrics
            </p>
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
          {/* RBAC Document-Level Access Control Card */}
          <div className="explain-section">
            <div className="explain-section-title">
              <ShieldCheck size={16} style={{ color: '#38bdf8' }} />
              Role-Based Access Control (RBAC) Security Boundary
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', marginTop: '0.5rem' }}>
              <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '0.75rem' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Active Role Tier</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#38bdf8', marginTop: '0.2rem' }}>
                  {activeSecurityRole}
                </div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '0.75rem' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Pre-Retrieval Filtered</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: blockedCount > 0 ? '#f43f5e' : '#10b981', marginTop: '0.2rem' }}>
                  {blockedCount > 0 ? `${blockedCount} Passages Blocked` : '0 Blocked (Full Access)'}
                </div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '0.75rem' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Authorized Ingested Chunks</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#34d399', marginTop: '0.2rem' }}>
                  {retrieval?.candidate_count !== undefined ? `${retrieval.candidate_count} Chunks` : 'Authorized'}
                </div>
              </div>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '0.6rem' }}>
              Unauthorized document chunks are purged at the pre-retrieval boundary. Neither the cross-encoder re-ranker nor LLM context window ever receives restricted content.
            </p>
          </div>

          {/* Section 1: RAGAS Quality Metrics */}
          <div className="explain-section">
            <div className="explain-section-title">
              <Gauge size={16} />
              RAGAS Production Evaluation Metrics
            </div>
            <div className="gauge-grid">
              <div className="gauge-card">
                <div className="gauge-label">Faithfulness (Groundedness)</div>
                <div className="gauge-value" style={{ color: '#10b981' }}>
                  {ragas.faithfulness !== undefined ? `${(ragas.faithfulness * 100).toFixed(1)}%` : 'N/A'}
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>
                  Grounded strictly in retrieved context
                </div>
              </div>
              <div className="gauge-card">
                <div className="gauge-label">Answer Relevancy</div>
                <div className="gauge-value" style={{ color: '#38bdf8' }}>
                  {ragas.relevancy !== undefined ? `${(ragas.relevancy * 100).toFixed(1)}%` : 'N/A'}
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>
                  Semantic fidelity to query intent
                </div>
              </div>
              <div className="gauge-card">
                <div className="gauge-label">Routing Confidence</div>
                <div className="gauge-value" style={{ color: '#a855f7' }}>
                  {intent?.confidence !== undefined ? `${(intent.confidence * 100).toFixed(1)}%` : 'N/A'}
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>
                  Intent Classifier certainty
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Agentic Intent Routing */}
          <div className="explain-section">
            <div className="explain-section-title">
              <ShieldAlert size={16} />
              Out-of-Distribution Intent Guardrail
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.6rem' }}>
              <span className={`intent-badge intent-${(intent?.intent || 'in_domain_rag').toLowerCase()}`}>
                ● {intent?.intent || 'IN_DOMAIN_RAG'}
              </span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                Decision latency: {intent?.latency_ms || latency?.intent_ms || 0} ms
              </span>
            </div>
            <p style={{ fontSize: '0.82rem', color: '#cbd5e1', background: 'rgba(0,0,0,0.25)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              {intent?.rationale || "Grounded query passed semantic domain boundaries."}
            </p>
          </div>

          {/* Section 3: Cross-Encoder Re-ranking Matrix */}
          {rerank?.top_chunks && rerank.top_chunks.length > 0 && (
            <div className="explain-section">
              <div className="explain-section-title">
                <Cpu size={16} />
                Cross-Encoder Re-Ranking Matrix (ms-marco-MiniLM-L-6-v2)
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: '0.75rem' }}>
                Shows how cross-attention between (query, passage) corrected the initial hybrid retrieval rank.
              </p>
              <table className="explain-table">
                <thead>
                  <tr>
                    <th>Doc & Chunk</th>
                    <th>Initial Rank</th>
                    <th>Final Rank</th>
                    <th>Delta</th>
                    <th>Relevance Score</th>
                    <th>Type</th>
                  </tr>
                </thead>
                <tbody>
                  {rerank.top_chunks.map((c, i) => (
                    <tr key={i}>
                      <td style={{ color: '#e2e8f0', fontWeight: 600 }}>
                        {c.document_filename} (Chunk #{c.chunk_index})
                      </td>
                      <td>#{c.initial_rank}</td>
                      <td style={{ color: 'var(--primary)', fontWeight: 600 }}>#{c.final_rank}</td>
                      <td>
                        {c.rank_delta > 0 ? (
                          <span className="delta-positive">
                            <ArrowUp size={12} style={{ display: 'inline', verticalAlign: 'middle' }} /> +{c.rank_delta}
                          </span>
                        ) : c.rank_delta < 0 ? (
                          <span className="delta-negative">
                            <ArrowDown size={12} style={{ display: 'inline', verticalAlign: 'middle' }} /> {c.rank_delta}
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-dim)' }}>
                            <Minus size={12} style={{ display: 'inline', verticalAlign: 'middle' }} /> 0
                          </span>
                        )}
                      </td>
                      <td style={{ color: '#34d399', fontWeight: 600 }}>
                        {(c.relevance_score * 100).toFixed(1)}%
                      </td>
                      <td>
                        {c.is_table ? (
                          <span style={{ background: 'rgba(6, 182, 212, 0.15)', color: '#06b6d4', padding: '0.15rem 0.4rem', borderRadius: '4px', fontSize: '0.7rem' }}>
                            TABLE
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>
                            TEXT
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Section 4: Hybrid Candidates (FAISS vs BM25) */}
          {retrieval?.candidates && retrieval.candidates.length > 0 && (
            <div className="explain-section">
              <div className="explain-section-title">
                <Database size={16} />
                Hybrid Candidate Pool (FAISS Dense + BM25 Sparse Fusion)
              </div>
              <table className="explain-table">
                <thead>
                  <tr>
                    <th>Chunk ID</th>
                    <th>Dense Rank (FAISS)</th>
                    <th>Sparse Rank (BM25)</th>
                    <th>RRF Score</th>
                    <th>Snippet</th>
                  </tr>
                </thead>
                <tbody>
                  {retrieval.candidates.map((c, idx) => (
                    <tr key={idx}>
                      <td style={{ color: 'var(--text-dim)' }}>#{c.chunk_id}</td>
                      <td>{c.dense_rank ? `#${c.dense_rank} (${(c.dense_similarity * 100).toFixed(0)}%)` : '—'}</td>
                      <td>{c.bm25_rank ? `#${c.bm25_rank} (${c.bm25_score.toFixed(1)})` : '—'}</td>
                      <td style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{c.rrf_score.toFixed(4)}</td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.72rem', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {c.snippet}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Section 5: Latency Waterfall */}
          <div className="explain-section">
            <div className="explain-section-title">
              <Clock size={16} />
              Latency Breakdown (Milliseconds)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.6rem', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>INTENT</div>
                <div style={{ fontSize: '1rem', fontWeight: 600 }}>{latency.intent_ms || 0} ms</div>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.6rem', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>RETRIEVAL</div>
                <div style={{ fontSize: '1rem', fontWeight: 600 }}>{latency.retrieval_ms || 0} ms</div>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.6rem', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>CROSS-ENCODER</div>
                <div style={{ fontSize: '1rem', fontWeight: 600 }}>{latency.rerank_ms || 0} ms</div>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.6rem', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>TIME TO FIRST TOKEN</div>
                <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>{latency.ttft_ms || 0} ms</div>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.6rem', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>TOTAL PIPELINE</div>
                <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--primary)' }}>{latency.total_ms || 0} ms</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
