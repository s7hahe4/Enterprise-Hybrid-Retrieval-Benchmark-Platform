import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Award, 
  TrendingUp, 
  Zap, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Download, 
  RefreshCw, 
  Play, 
  FileText, 
  ChevronDown, 
  ChevronUp, 
  Layers, 
  HelpCircle,
  Database,
  Cpu,
  Eye,
  Sliders
} from 'lucide-react';

const CONFIG_COLORS = {
  'FAISS_ONLY': '#38bdf8',               // cyan
  'BM25_ONLY': '#f59e0b',                // amber
  'NAIVE_HYBRID': '#64748b',             // slate
  'HYBRID_RRF': '#10b981',               // emerald
  'HYBRID_RRF_CROSS_ENCODER': '#a855f7'  // purple
};

const CONFIG_BADGES = {
  'FAISS_ONLY': 'Dense Vector',
  'BM25_ONLY': 'Sparse Lexical',
  'NAIVE_HYBRID': 'Naive Linear',
  'HYBRID_RRF': 'RRF Fusion',
  'HYBRID_RRF_CROSS_ENCODER': 'Cross-Encoder Re-ranked'
};

export default function BenchmarkDashboard() {
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [historyRuns, setHistoryRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [questionCount, setQuestionCount] = useState(6);
  const [sortKey, setSortKey] = useState('mean_mrr');
  const [sortAsc, setSortAsc] = useState(false);
  const [expandedQueries, setExpandedQueries] = useState({});
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [syntheticPreview, setSyntheticPreview] = useState([]);
  const [isGeneratingDataset, setIsGeneratingDataset] = useState(false);

  // Fetch benchmark history on mount
  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/documents/rag/benchmark/runs/');
      if (res.ok) {
        const runs = await res.json();
        setHistoryRuns(runs);
        if (runs.length > 0 && !benchmarkData) {
          fetchRunDetail(runs[0].id);
        }
      }
    } catch (err) {
      console.error('Failed to load benchmark history', err);
    }
  };

  const fetchRunDetail = async (runId) => {
    try {
      const res = await fetch(`/api/documents/rag/benchmark/runs/${runId}/`);
      if (res.ok) {
        const data = await res.json();
        setBenchmarkData(data);
        setSelectedRunId(runId);
      }
    } catch (err) {
      console.error('Failed to load benchmark run details', err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // Run new benchmark experiment
  const handleRunBenchmark = async () => {
    setIsRunning(true);
    try {
      const res = await fetch('/api/documents/rag/benchmark/run/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_questions: questionCount,
          name: `Benchmark Run (${questionCount} Queries)`
        })
      });

      if (res.ok) {
        const result = await res.json();
        setBenchmarkData(result);
        if (result.benchmark_run_id) {
          setSelectedRunId(result.benchmark_run_id);
          fetchHistory();
        }
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Benchmark Error: ${err.error || 'Failed to run benchmark suite'}`);
      }
    } catch (err) {
      alert(`Benchmark Error: ${err.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  // Preview synthetic test set
  const handleGenerateDataset = async () => {
    setIsGeneratingDataset(true);
    try {
      const res = await fetch('/api/documents/rag/benchmark/generate-dataset/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_questions: questionCount })
      });
      if (res.ok) {
        const data = await res.json();
        setSyntheticPreview(data.test_cases || []);
        setShowDatasetModal(true);
      }
    } catch (err) {
      console.error('Failed to generate test set preview', err);
    } finally {
      setIsGeneratingDataset(false);
    }
  };

  // Toggle accordion query
  const toggleQuery = (qid) => {
    setExpandedQueries(prev => ({
      ...prev,
      [qid]: !prev[qid]
    }));
  };

  // Export to CSV
  const exportCSV = () => {
    if (!benchmarkData?.summary_metrics) return;
    const headers = ['Configuration', 'Recall@1', 'Recall@3', 'Recall@5', 'MRR', 'NDCG@5', 'Precision@3', 'Avg Latency (ms)', 'MRR Lift (%)'];
    const rows = Object.values(benchmarkData.summary_metrics).map(m => [
      `"${m.display_name}"`,
      m.recall_1,
      m.recall_3,
      m.recall_5,
      m.mean_mrr,
      m.mean_ndcg_5,
      m.precision_3,
      m.avg_latency_ms,
      m.mrr_lift_pct
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `rag_benchmark_summary_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export to JSON
  const exportJSON = () => {
    if (!benchmarkData) return;
    const blob = new Blob([JSON.stringify(benchmarkData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `rag_benchmark_experiment_${Date.now()}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Summary list sorted
  const summaryList = benchmarkData?.summary_metrics
    ? Object.values(benchmarkData.summary_metrics).sort((a, b) => {
        const valA = a[sortKey] ?? 0;
        const valB = b[sortKey] ?? 0;
        return sortAsc ? valA - valB : valB - valA;
      })
    : [];

  // Determine top performer
  const topPerformer = summaryList.length > 0 
    ? [...summaryList].sort((a, b) => (b.mean_mrr + b.mean_ndcg_5) - (a.mean_mrr + a.mean_ndcg_5))[0]
    : null;

  const crossEncoderStats = benchmarkData?.summary_metrics?.['HYBRID_RRF_CROSS_ENCODER'];
  const faissStats = benchmarkData?.summary_metrics?.['FAISS_ONLY'];

  return (
    <div className="benchmark-container">
      {/* Top Header & Control Bar */}
      <div className="benchmark-header">
        <div className="benchmark-title-group">
          <div className="benchmark-badge">
            <BarChart3 size={16} />
            <span>Empirical Evaluation Lab</span>
          </div>
          <h2>RAG Retrieval Benchmark Leaderboard</h2>
          <p>
            Rigorous evaluation comparing <strong>FAISS (Dense)</strong>, <strong>BM25 (Sparse)</strong>, 
            <strong>Hybrid RRF</strong>, and <strong>Cross-Encoder Re-ranking</strong> across standard Information Retrieval metrics.
          </p>
        </div>

        <div className="benchmark-actions">
          {/* History selector */}
          {historyRuns.length > 0 && (
            <div className="history-selector-wrapper">
              <label>Experiment History:</label>
              <select 
                value={selectedRunId} 
                onChange={(e) => fetchRunDetail(e.target.value)}
                className="benchmark-select"
              >
                {historyRuns.map(run => (
                  <option key={run.id} value={run.id}>
                    {run.name} ({new Date(run.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })})
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Question count control */}
          <div className="count-selector-wrapper">
            <label>Queries:</label>
            <select 
              value={questionCount} 
              onChange={(e) => setQuestionCount(Number(e.target.value))}
              className="benchmark-select"
              disabled={isRunning}
            >
              <option value={4}>4 Queries (Fast)</option>
              <option value={6}>6 Queries (Standard)</option>
              <option value={8}>8 Queries (Deep)</option>
              <option value={12}>12 Queries (Extensive)</option>
            </select>
          </div>

          <button 
            onClick={handleGenerateDataset}
            className="secondary-btn"
            disabled={isRunning || isGeneratingDataset}
            title="Inspect or preview synthetic questions"
          >
            <Eye size={15} />
            <span>Preview Dataset</span>
          </button>

          <button 
            onClick={handleRunBenchmark}
            className="run-benchmark-btn"
            disabled={isRunning}
          >
            {isRunning ? (
              <>
                <RefreshCw size={16} className="spin" />
                <span>Running Benchmark Suite...</span>
              </>
            ) : (
              <>
                <Play size={16} />
                <span>Execute Benchmark</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* KPI Cards Row */}
      {benchmarkData && (
        <div className="kpi-cards-grid">
          <div className="kpi-card highlight-card">
            <div className="kpi-icon-wrapper crown-icon">
              <Award size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-label">Champion Architecture</span>
              <h4 className="kpi-value">{topPerformer?.display_name.split(' (')[0] || 'Cross-Encoder'}</h4>
              <span className="kpi-subtext">Highest combined MRR ({topPerformer?.mean_mrr}) & NDCG@5 ({topPerformer?.mean_ndcg_5})</span>
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-icon-wrapper lift-icon">
              <TrendingUp size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-label">Cross-Encoder MRR Lift</span>
              <h4 className="kpi-value">
                {crossEncoderStats ? `+${crossEncoderStats.mrr_lift_pct}%` : 'N/A'}
              </h4>
              <span className="kpi-subtext">Ranking gain over single dense vector retrieval</span>
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-icon-wrapper recall-icon">
              <Layers size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-label">Peak Recall@5</span>
              <h4 className="kpi-value">
                {topPerformer ? `${(topPerformer.recall_5 * 100).toFixed(1)}%` : '0%'}
              </h4>
              <span className="kpi-subtext">Target passage retrieved within top 5 candidates</span>
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-icon-wrapper speed-icon">
              <Zap size={22} />
            </div>
            <div className="kpi-content">
              <span className="kpi-label">Re-ranking Latency</span>
              <h4 className="kpi-value">
                {crossEncoderStats ? `${crossEncoderStats.avg_latency_ms}ms` : '0ms'}
              </h4>
              <span className="kpi-subtext">Avg latency per query across joint attention</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Leaderboard Section */}
      <div className="benchmark-card">
        <div className="benchmark-card-header">
          <div className="card-title-group">
            <h3>Retrieval Configuration Leaderboard</h3>
            <span className="meta-badge">{benchmarkData?.total_queries || 0} Test Queries Evaluated</span>
          </div>

          <div className="export-buttons-group">
            <button onClick={exportCSV} className="export-btn" title="Export Leaderboard to CSV">
              <Download size={14} />
              <span>Export CSV</span>
            </button>
            <button onClick={exportJSON} className="export-btn" title="Export Full Report to JSON">
              <FileText size={14} />
              <span>Export JSON</span>
            </button>
          </div>
        </div>

        {/* Leaderboard Table */}
        <div className="table-responsive">
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th style={{ textAlign: 'left', minWidth: '220px' }}>Retrieval Strategy</th>
                <th onClick={() => { setSortKey('recall_1'); setSortAsc(!sortAsc); }} className="sortable-th">
                  Recall@1 {sortKey === 'recall_1' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => { setSortKey('recall_3'); setSortAsc(!sortAsc); }} className="sortable-th">
                  Recall@3 {sortKey === 'recall_3' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => { setSortKey('recall_5'); setSortAsc(!sortAsc); }} className="sortable-th">
                  Recall@5 {sortKey === 'recall_5' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => { setSortKey('mean_mrr'); setSortAsc(!sortAsc); }} className="sortable-th">
                  MRR {sortKey === 'mean_mrr' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => { setSortKey('mean_ndcg_5'); setSortAsc(!sortAsc); }} className="sortable-th">
                  NDCG@5 {sortKey === 'mean_ndcg_5' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => { setSortKey('precision_3'); setSortAsc(!sortAsc); }} className="sortable-th">
                  Precision@3 {sortKey === 'precision_3' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => { setSortKey('avg_latency_ms'); setSortAsc(!sortAsc); }} className="sortable-th">
                  Latency {sortKey === 'avg_latency_ms' && (sortAsc ? '▲' : '▼')}
                </th>
                <th style={{ textAlign: 'right' }}>MRR Lift</th>
              </tr>
            </thead>
            <tbody>
              {summaryList.map((cfg) => {
                const isChampion = topPerformer?.config_id === cfg.config_id;
                const color = CONFIG_COLORS[cfg.config_id] || '#94a3b8';
                return (
                  <tr key={cfg.config_id} className={isChampion ? 'champion-row' : ''}>
                    <td>
                      <div className="strategy-cell">
                        <span className="strategy-dot" style={{ backgroundColor: color }}></span>
                        <div>
                          <div className="strategy-name">
                            {cfg.display_name.split(' (')[0]}
                            {isChampion && <span className="champion-tag">Top Performer</span>}
                          </div>
                          <span className="strategy-subtitle">{CONFIG_BADGES[cfg.config_id]}</span>
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="metric-cell">
                        <span className="metric-number">{(cfg.recall_1 * 100).toFixed(1)}%</span>
                        <div className="metric-bar-bg">
                          <div className="metric-bar-fill" style={{ width: `${cfg.recall_1 * 100}%`, backgroundColor: color }}></div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="metric-cell">
                        <span className="metric-number">{(cfg.recall_3 * 100).toFixed(1)}%</span>
                        <div className="metric-bar-bg">
                          <div className="metric-bar-fill" style={{ width: `${cfg.recall_3 * 100}%`, backgroundColor: color }}></div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="metric-cell">
                        <span className="metric-number">{(cfg.recall_5 * 100).toFixed(1)}%</span>
                        <div className="metric-bar-bg">
                          <div className="metric-bar-fill" style={{ width: `${cfg.recall_5 * 100}%`, backgroundColor: color }}></div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="metric-highlight">{cfg.mean_mrr.toFixed(3)}</span>
                    </td>
                    <td>
                      <span className="metric-highlight">{cfg.mean_ndcg_5.toFixed(3)}</span>
                    </td>
                    <td>
                      <span>{(cfg.precision_3 * 100).toFixed(1)}%</span>
                    </td>
                    <td>
                      <div className="latency-pill-cell">
                        <Clock size={12} />
                        <span>{cfg.avg_latency_ms} ms</span>
                      </div>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      {cfg.config_id === 'FAISS_ONLY' ? (
                        <span className="baseline-tag">Baseline</span>
                      ) : (
                        <span className={`lift-tag ${cfg.mrr_lift_pct >= 0 ? 'positive' : 'negative'}`}>
                          {cfg.mrr_lift_pct >= 0 ? `+${cfg.mrr_lift_pct}%` : `${cfg.mrr_lift_pct}%`}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Comparative Charts */}
      {benchmarkData && (
        <div className="charts-grid">
          {/* Chart 1: MRR Comparison */}
          <div className="benchmark-card chart-card">
            <div className="card-title-group">
              <h4>MRR (Mean Reciprocal Rank)</h4>
              <span className="chart-legend">Higher is better (1.0 = Perfect First Hit)</span>
            </div>
            <div className="bar-chart-vertical">
              {summaryList.map(cfg => {
                const color = CONFIG_COLORS[cfg.config_id] || '#94a3b8';
                const heightPct = Math.max(8, cfg.mean_mrr * 100);
                return (
                  <div key={cfg.config_id} className="chart-bar-column">
                    <span className="bar-val">{cfg.mean_mrr.toFixed(3)}</span>
                    <div className="bar-wrapper">
                      <div 
                        className="bar-rect" 
                        style={{ height: `${heightPct}%`, backgroundColor: color }}
                        title={`${cfg.display_name}: ${cfg.mean_mrr}`}
                      ></div>
                    </div>
                    <span className="bar-label">{CONFIG_BADGES[cfg.config_id]?.split(' ')[0]}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Chart 2: NDCG@5 Comparison */}
          <div className="benchmark-card chart-card">
            <div className="card-title-group">
              <h4>NDCG@5 (Normalized Discounted Gain)</h4>
              <span className="chart-legend">Measures ranking order quality</span>
            </div>
            <div className="bar-chart-vertical">
              {summaryList.map(cfg => {
                const color = CONFIG_COLORS[cfg.config_id] || '#94a3b8';
                const heightPct = Math.max(8, cfg.mean_ndcg_5 * 100);
                return (
                  <div key={cfg.config_id} className="chart-bar-column">
                    <span className="bar-val">{cfg.mean_ndcg_5.toFixed(3)}</span>
                    <div className="bar-wrapper">
                      <div 
                        className="bar-rect" 
                        style={{ height: `${heightPct}%`, backgroundColor: color }}
                        title={`${cfg.display_name}: ${cfg.mean_ndcg_5}`}
                      ></div>
                    </div>
                    <span className="bar-label">{CONFIG_BADGES[cfg.config_id]?.split(' ')[0]}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Chart 3: Latency Trade-off */}
          <div className="benchmark-card chart-card">
            <div className="card-title-group">
              <h4>Execution Latency (ms)</h4>
              <span className="chart-legend">Dense vs Joint-Attention Re-ranking</span>
            </div>
            <div className="bar-chart-vertical">
              {summaryList.map(cfg => {
                const maxLat = Math.max(...summaryList.map(s => s.avg_latency_ms), 1);
                const heightPct = Math.max(10, (cfg.avg_latency_ms / maxLat) * 100);
                const color = CONFIG_COLORS[cfg.config_id] || '#94a3b8';
                return (
                  <div key={cfg.config_id} className="chart-bar-column">
                    <span className="bar-val">{cfg.avg_latency_ms}ms</span>
                    <div className="bar-wrapper">
                      <div 
                        className="bar-rect" 
                        style={{ height: `${heightPct}%`, backgroundColor: color }}
                        title={`${cfg.display_name}: ${cfg.avg_latency_ms}ms`}
                      ></div>
                    </div>
                    <span className="bar-label">{CONFIG_BADGES[cfg.config_id]?.split(' ')[0]}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Per-Query Diagnostic Drilldown */}
      {benchmarkData?.detailed_results && benchmarkData.detailed_results.length > 0 && (
        <div className="benchmark-card drilldown-card">
          <div className="benchmark-card-header">
            <div className="card-title-group">
              <h3>Per-Query Diagnostic Drilldown</h3>
              <span className="meta-badge">Inspect exact rank positions per retriever</span>
            </div>
          </div>

          <div className="query-accordion-list">
            {benchmarkData.detailed_results.map((item, idx) => {
              const isExpanded = !!expandedQueries[item.id];
              return (
                <div key={item.id} className="query-accordion-item">
                  <div className="query-accordion-header" onClick={() => toggleQuery(item.id)}>
                    <div className="query-header-left">
                      <span className="query-idx">#{idx + 1}</span>
                      <div>
                        <div className="query-text">"{item.query}"</div>
                        <div className="query-meta-chips">
                          <span className="meta-chip doc-chip">{item.document_filename || 'PDF'}</span>
                          {item.target_heading && (
                            <span className="meta-chip section-chip">§ {item.target_heading}</span>
                          )}
                          <span className="meta-chip target-chip">Target Chunk ID: {item.relevant_chunk_ids?.join(', ')}</span>
                        </div>
                      </div>
                    </div>

                    <div className="query-header-right">
                      {/* Compact status pill for Cross-Encoder */}
                      {item.config_evaluations?.HYBRID_RRF_CROSS_ENCODER && (
                        <span className={`status-pill ${item.config_evaluations.HYBRID_RRF_CROSS_ENCODER.hit_within_k ? 'hit' : 'miss'}`}>
                          CE: {item.config_evaluations.HYBRID_RRF_CROSS_ENCODER.first_hit_rank ? `Rank #${item.config_evaluations.HYBRID_RRF_CROSS_ENCODER.first_hit_rank}` : 'Missed'}
                        </span>
                      )}
                      {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="query-accordion-body">
                      {/* Passage Excerpt */}
                      {item.passage_excerpt && (
                        <div className="ground-truth-box">
                          <div className="gt-title">Ground Truth Target Passage:</div>
                          <p className="gt-text">"{item.passage_excerpt}"</p>
                        </div>
                      )}

                      {/* Config comparison cards for this query */}
                      <div className="config-grid-row">
                        {Object.entries(item.config_evaluations || {}).map(([cfgKey, evalData]) => {
                          const color = CONFIG_COLORS[cfgKey] || '#94a3b8';
                          const isHit = evalData.hit_within_k;
                          const hitRank = evalData.first_hit_rank;
                          return (
                            <div key={cfgKey} className={`config-eval-card ${isHit ? 'eval-success' : 'eval-failed'}`}>
                              <div className="eval-card-header">
                                <span className="eval-badge" style={{ borderColor: color, color: color }}>
                                  {CONFIG_BADGES[cfgKey] || cfgKey}
                                </span>
                                {isHit ? (
                                  <span className="hit-indicator success">
                                    <CheckCircle size={14} /> #{hitRank}
                                  </span>
                                ) : (
                                  <span className="hit-indicator fail">
                                    <XCircle size={14} /> Miss
                                  </span>
                                )}
                              </div>

                              <div className="eval-metrics-row">
                                <div className="eval-stat">
                                  <span className="eval-stat-label">MRR</span>
                                  <span className="eval-stat-value">{evalData.mrr}</span>
                                </div>
                                <div className="eval-stat">
                                  <span className="eval-stat-label">NDCG@5</span>
                                  <span className="eval-stat-value">{evalData.ndcg_5}</span>
                                </div>
                                <div className="eval-stat">
                                  <span className="eval-stat-label">Latency</span>
                                  <span className="eval-stat-value">{evalData.latency_ms}ms</span>
                                </div>
                              </div>

                              <div className="retrieved-ids-row">
                                <span className="retrieved-label">Top Chunks:</span>
                                <span className="retrieved-ids">
                                  {evalData.retrieved_chunk_ids?.length > 0 
                                    ? evalData.retrieved_chunk_ids.map(id => (
                                        <span 
                                          key={id} 
                                          className={`id-tag ${item.relevant_chunk_ids?.includes(id) ? 'target-id' : ''}`}
                                        >
                                          {id}
                                        </span>
                                      ))
                                    : 'None'}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Dataset Preview Modal */}
      {showDatasetModal && (
        <div className="modal-backdrop" onClick={() => setShowDatasetModal(false)}>
          <div className="modal-container dataset-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <h3>Synthetic Evaluation Test Suite</h3>
                <span className="meta-badge">{syntheticPreview.length} Queries Generated</span>
              </div>
              <button className="modal-close-btn" onClick={() => setShowDatasetModal(false)}>✕</button>
            </div>

            <div className="modal-body">
              <p className="modal-desc">
                These ground-truth test pairs were automatically extracted from your ingested PDF chunks
                using heading detection and salient entity heuristics:
              </p>
              <div className="dataset-list">
                {syntheticPreview.map((tc, i) => (
                  <div key={i} className="dataset-card">
                    <div className="dataset-card-header">
                      <span className="query-idx">#{i + 1}</span>
                      <span className="dataset-query-text">"{tc.query}"</span>
                    </div>
                    <div className="dataset-card-meta">
                      <span>Doc: {tc.document_filename}</span>
                      <span>§ {tc.target_heading}</span>
                      <span>Target Chunk: #{tc.relevant_chunk_ids?.[0]}</span>
                    </div>
                    {tc.passage_excerpt && (
                      <div className="dataset-excerpt">"{tc.passage_excerpt}"</div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="modal-footer">
              <button className="secondary-btn" onClick={() => setShowDatasetModal(false)}>Close</button>
              <button 
                className="run-benchmark-btn" 
                onClick={() => {
                  setShowDatasetModal(false);
                  handleRunBenchmark();
                }}
              >
                <Play size={15} />
                <span>Run Benchmark With This Test Set</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
