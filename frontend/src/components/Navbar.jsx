import React from 'react';
import { Cpu, Database, Sparkles, Activity, MessageSquare, BarChart3, Shield, ShieldAlert, ShieldCheck } from 'lucide-react';

export default function Navbar({ latency, activeTab, setActiveTab, userRole = 'PUBLIC', setUserRole }) {
  const ROLES = [
    { key: 'PUBLIC', label: 'Public Tier', icon: '🌐', color: '#38bdf8' },
    { key: 'ENGINEERING', label: 'Engineering', icon: '⚙️', color: '#818cf8' },
    { key: 'HR', label: 'Human Resources', icon: '👥', color: '#f472b6' },
    { key: 'FINANCE', label: 'Finance & Legal', icon: '💰', color: '#34d399' },
    { key: 'ADMIN', label: 'Admin (Unrestricted)', icon: '👑', color: '#f59e0b' }
  ];

  const currentRoleObj = ROLES.find(r => r.key === userRole) || ROLES[0];

  return (
    <nav className="navbar">
      <div className="nav-brand">
        <div className="brand-icon">
          <Sparkles size={20} />
        </div>
        <div>
          <h1 className="brand-title">RAG Intelligence Platform</h1>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', letterSpacing: '0.04em' }}>
            Enterprise Hybrid Vector, Cross-Encoder & RBAC Architecture
          </span>
        </div>
      </div>

      <div className="nav-tabs">
        <button 
          className={`nav-tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          <MessageSquare size={15} />
          <span>RAG Copilot</span>
        </button>
        <button 
          className={`nav-tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          <BarChart3 size={15} />
          <span>Benchmark Lab</span>
          <span className="tab-pill-badge">Phase 2</span>
        </button>
      </div>

      <div className="status-pills">
        {/* RBAC Role Switcher */}
        <div className="role-switcher-pill" title="Current security role tier for retrieval authorization">
          <Shield size={13} style={{ color: currentRoleObj.color }} />
          <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>Role:</span>
          <select 
            value={userRole} 
            onChange={(e) => setUserRole && setUserRole(e.target.value)}
            className="role-select"
            style={{ color: currentRoleObj.color }}
          >
            {ROLES.map(r => (
              <option key={r.key} value={r.key}>
                {r.icon} {r.label}
              </option>
            ))}
          </select>
        </div>

        <div className="pill">
          <span className="pill-dot"></span>
          <Database size={13} style={{ color: 'var(--accent-cyan)' }} />
          <span>FAISS + BM25</span>
        </div>
        <div className="pill">
          <Cpu size={13} style={{ color: '#a855f7' }} />
          <span>Cross-Encoder</span>
        </div>
        <div className="pill">
          <Activity size={13} style={{ color: '#10b981' }} />
          <span>Latency: {latency ? `${latency}ms` : 'Ready'}</span>
        </div>
      </div>
    </nav>
  );
}


