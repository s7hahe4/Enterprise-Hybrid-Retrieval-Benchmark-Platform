import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileText, 
  CheckCircle2, 
  RefreshCw, 
  Zap, 
  GitCompare, 
  Layers,
  Clock,
  Shield,
  Lock,
  Trash2
} from 'lucide-react';

const ROLE_CONFIGS = {
  'PUBLIC': { label: 'Public', icon: '🌐', color: '#38bdf8', bg: 'rgba(56, 189, 248, 0.12)' },
  'ENGINEERING': { label: 'Engineering', icon: '⚙️', color: '#818cf8', bg: 'rgba(129, 140, 248, 0.12)' },
  'HR': { label: 'HR', icon: '👥', color: '#f472b6', bg: 'rgba(244, 114, 182, 0.12)' },
  'FINANCE': { label: 'Finance', icon: '💰', color: '#34d399', bg: 'rgba(52, 211, 153, 0.12)' },
  'ADMIN': { label: 'Admin', icon: '👑', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.12)' }
};

export default function Sidebar({ 
  documents, 
  loadingDocs, 
  uploading, 
  activeIngestJob,
  userRole = 'PUBLIC',
  onUploadFile, 
  onUpdateDocRole,
  onDeleteDoc,
  onRefreshDocs, 
  onSelectPrompt,
  onOpenCompare 
}) {
  const fileInputRef = useRef(null);
  const [uploadRole, setUploadRole] = useState('PUBLIC');

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUploadFile(e.target.files[0], uploadRole);
    }
  };

  const PRESETS = [
    {
      label: "Ask About ML & Experience",
      badge: "KNOWLEDGE",
      query: "What machine learning and backend engineering experience does Shahedul have?",
      type: "rag"
    },
    {
      label: "Ask Confidential Payroll",
      badge: "SECURITY",
      query: "What is the confidential payroll ledger and executive bonus package?",
      type: "rbac"
    },
    {
      label: "Ask Technical Research & Stats",
      badge: "DEEP SEARCH",
      query: "What research did he perform on Monte Carlo Dropout and Wilcoxon tests?",
      type: "rag"
    },
    {
      label: "Say Hello / Capabilities",
      badge: "CHITCHAT",
      query: "Hello! What can you do?",
      type: "chitchat"
    },
    {
      label: "Ask Off-Topic (Baking Recipe)",
      badge: "GUARDRAIL",
      query: "What is the best recipe for baking chocolate cookies?",
      type: "ood"
    }
  ];

  return (
    <aside className="sidebar">
      {/* Upload Box & Live Progress Bar with RBAC classification */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
          <h2 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>
            Document Ingestion
          </h2>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>
            Tier: <strong style={{ color: ROLE_CONFIGS[uploadRole]?.color }}>{uploadRole}</strong>
          </span>
        </div>

        {/* Target Security Tier Selector */}
        <div className="upload-role-bar">
          <Shield size={12} style={{ color: ROLE_CONFIGS[uploadRole]?.color }} />
          <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Upload As:</span>
          <select 
            value={uploadRole} 
            onChange={(e) => setUploadRole(e.target.value)}
            className="upload-role-dropdown"
          >
            <option value="PUBLIC">🌐 Public (All Users)</option>
            <option value="ENGINEERING">⚙️ Engineering Tier</option>
            <option value="HR">👥 HR (Confidential)</option>
            <option value="FINANCE">💰 Finance (Restricted)</option>
            <option value="ADMIN">👑 Administrator Only</option>
          </select>
        </div>
        
        {activeIngestJob ? (
          <div className="active-ingest-card">
            <div className="ingest-card-header">
              <div className="ingest-filename">
                <FileText size={15} style={{ color: 'var(--accent-cyan)' }} />
                <span>{activeIngestJob.filename}</span>
              </div>
              <span className="ingest-pct-badge">{activeIngestJob.progress_pct || 10}%</span>
            </div>

            <div className="ingest-progress-track">
              <div 
                className="ingest-progress-fill" 
                style={{ width: `${Math.max(8, activeIngestJob.progress_pct || 10)}%` }}
              ></div>
            </div>

            <div className="ingest-step-desc">
              <RefreshCw size={11} className="spin" />
              <span>{activeIngestJob.current_step || 'Processing PDF...'}</span>
            </div>
          </div>
        ) : (
          <div 
            className="sidebar-dropzone"
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            <UploadCloud size={28} style={{ color: 'var(--primary)' }} />
            <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>
              {uploading ? 'Uploading PDF...' : 'Upload PDF Document'}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
              Auto-assigned to <strong>{ROLE_CONFIGS[uploadRole]?.label}</strong> security tier
            </div>
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept=".pdf" 
              onChange={handleFileChange}
              disabled={uploading || !!activeIngestJob}
            />
          </div>
        )}
      </div>


      {/* Preset Prompts */}
      <div>
        <div style={{ marginBottom: '0.6rem' }}>
          <h2 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Zap size={14} style={{ color: '#f59e0b' }} />
            Try Example Questions
          </h2>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', display: 'block', marginTop: '0.2rem' }}>
            Click an example to test AI routing & security
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              onClick={() => onSelectPrompt(p.query)}
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid var(--border-subtle)',
                color: '#cbd5e1',
                borderRadius: '8px',
                padding: '0.55rem 0.75rem',
                fontSize: '0.78rem',
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--primary)';
                e.currentTarget.style.background = 'rgba(99, 102, 241, 0.12)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-subtle)';
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
              }}
            >
              <span>{p.label}</span>
              <span style={{ 
                fontSize: '0.62rem', 
                padding: '0.15rem 0.45rem', 
                borderRadius: '4px',
                background: p.type === 'rag' ? 'rgba(16, 185, 129, 0.15)' : p.type === 'rbac' ? 'rgba(244, 114, 182, 0.2)' : p.type === 'chitchat' ? 'rgba(6, 182, 212, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                color: p.type === 'rag' ? '#34d399' : p.type === 'rbac' ? '#f472b6' : p.type === 'chitchat' ? '#38bdf8' : '#f87171',
                textTransform: 'uppercase',
                fontWeight: 600,
                letterSpacing: '0.03em'
              }}>
                {p.badge || p.type}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Ingested Documents List */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
          <h2 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Knowledge Base ({documents.length})
          </h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            {documents.length >= 2 && (
              <button 
                className="compare-versions-btn"
                onClick={() => onOpenCompare && onOpenCompare(documents[0], documents[1])}
                title="Compare document versions with semantic diffing"
              >
                <GitCompare size={12} />
                <span>Diff</span>
              </button>
            )}
            <button 
              onClick={onRefreshDocs}
              style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer' }}
              title="Refresh documents"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', overflowY: 'auto', flex: 1 }}>
          {loadingDocs ? (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', textAlign: 'center', padding: '1rem' }}>
              Loading documents...
            </div>
          ) : documents.length === 0 ? (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', textAlign: 'center', padding: '1rem' }}>
              No PDFs indexed yet. Upload one above.
            </div>
          ) : (
            documents.map((doc) => {
              const roleKey = (doc.access_role || 'PUBLIC').toUpperCase();
              const roleCfg = ROLE_CONFIGS[roleKey] || ROLE_CONFIGS['PUBLIC'];

              return (
                <div key={doc.id} className="doc-item">
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', overflow: 'hidden', flex: 1 }}>
                    <FileText size={16} style={{ color: 'var(--primary)', flexShrink: 0, marginTop: '2px' }} />
                    <div style={{ overflow: 'hidden', flex: 1 }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
                        <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '140px' }} title={doc.filename}>
                          {doc.filename}
                        </span>
                        {doc.version_number && (
                          <span className="doc-version-badge">v{doc.version_number}</span>
                        )}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.2rem', flexWrap: 'wrap' }}>
                        {/* Interactive Role Reclassification Pill */}
                        <div className="doc-role-wrapper">
                          <select 
                            value={roleKey}
                            onChange={(e) => onUpdateDocRole && onUpdateDocRole(doc.id, e.target.value)}
                            className="doc-role-select"
                            style={{ 
                              color: roleCfg.color, 
                              background: roleCfg.bg,
                              borderColor: `${roleCfg.color}44`
                            }}
                            title="Click to reclassify document security role"
                          >
                            <option value="PUBLIC">🌐 Public</option>
                            <option value="ENGINEERING">⚙️ Engineering</option>
                            <option value="HR">👥 HR</option>
                            <option value="FINANCE">💰 Finance</option>
                            <option value="ADMIN">👑 Admin</option>
                          </select>
                        </div>

                        <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>
                          {doc.chunks_count} chunks
                        </span>
                      </div>
                    </div>
                  </div>
                  <div style={{ flexShrink: 0, alignSelf: 'center', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <CheckCircle2 size={14} style={{ color: 'var(--success)' }} />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm(`Delete "${doc.filename}" from knowledge base?`)) {
                          onDeleteDoc && onDeleteDoc(doc.id, doc.filename);
                        }
                      }}
                      className="doc-delete-btn"
                      title="Delete document"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}


