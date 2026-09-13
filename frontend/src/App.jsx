import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ExplainabilityModal from './components/ExplainabilityModal';
import CitationModal from './components/CitationModal';
import BenchmarkDashboard from './components/BenchmarkDashboard';
import VersionDiffModal from './components/VersionDiffModal';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'benchmark'
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [activeIngestJob, setActiveIngestJob] = useState(null);
  const [diffModalDocs, setDiffModalDocs] = useState(null);
  
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [userRole, setUserRole] = useState('PUBLIC');
  const [latestLatency, setLatestLatency] = useState(null);

  const [explainModalData, setExplainModalData] = useState(null);
  const [selectedCitation, setSelectedCitation] = useState(null);

  // Fetch documents on load
  const fetchDocuments = async () => {
    setLoadingDocs(true);
    try {
      const res = await fetch('/api/documents/');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (err) {
      console.error('Failed to fetch documents', err);
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  // Update access control role for a document
  const handleUpdateDocumentRole = async (docId, newRole) => {
    try {
      const res = await fetch(`/api/documents/${docId}/role/`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ access_role: newRole })
      });
      if (res.ok) {
        fetchDocuments();
      }
    } catch (err) {
      console.error('Failed to update document role', err);
    }
  };

  // Async Background Upload PDF handler with Live Polling
  const handleUploadFile = async (file, accessRole = 'PUBLIC') => {
    if (!file) return;
    setUploading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('access_role', accessRole);

    try {
      const res = await fetch('/api/documents/upload/', {
        method: 'POST',
        body: formData
      });

      if (res.status === 202) {
        // 202 Accepted: Asynchronous Background Ingestion
        const jobData = await res.json();
        setActiveIngestJob({
          id: jobData.job_id,
          filename: jobData.filename,
          progress_pct: jobData.progress_pct || 10,
          current_step: jobData.current_step || 'Initializing parser...'
        });

        // Start live polling interval (every 800ms)
        const intervalId = setInterval(async () => {
          try {
            const jobRes = await fetch(`/api/documents/jobs/${jobData.job_id}/`);
            if (jobRes.ok) {
              const liveJob = await jobRes.json();
              setActiveIngestJob(liveJob);

              if (liveJob.status === 'completed') {
                clearInterval(intervalId);
                setActiveIngestJob(null);
                fetchDocuments();
                setMessages(prev => [
                  ...prev,
                  {
                    role: 'assistant',
                    intent: 'IN_DOMAIN_RAG',
                    confidence: 1.0,
                    content: `📄 **Document Ingested Successfully:** "${liveJob.filename}" is now fully parsed with **Parent-Child Chunking**, **Markdown Table Ingestion**, and **FAISS 384d Embeddings** [Tier: ${accessRole}].`
                  }
                ]);
              } else if (liveJob.status === 'failed') {
                clearInterval(intervalId);
                setActiveIngestJob(null);
                alert(`Ingestion failed: ${liveJob.error_message || 'Unknown processing error'}`);
              }
            }
          } catch (pollErr) {
            console.error('Error polling job status', pollErr);
          }
        }, 800);

      } else if (res.ok) {
        // Synchronous fallback
        const newDoc = await res.json();
        setDocuments(prev => [newDoc, ...prev.filter(d => d.id !== newDoc.id)]);
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            intent: 'IN_DOMAIN_RAG',
            confidence: 1.0,
            content: `📄 **Document Ingested Successfully:** "${newDoc.filename}" is ready.`
          }
        ]);
      } else {
        const errorData = await res.json();
        alert(`Upload error: ${errorData.error || 'Failed to upload'}`);
      }
    } catch (err) {
      alert(`Upload error: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };


  // Real-time SSE Streaming Query Handler with Conversational Memory & RBAC
  const handleSendMessage = async (queryText) => {
    if (!queryText.trim() || isStreaming) return;

    // 1. Prepare conversation history for multi-turn query rewriting
    const historyPayload = messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(-6)
      .map(m => ({ role: m.role, content: m.content }));

    // 2. Append user message
    const userMsg = { role: 'user', content: queryText };
    setMessages(prev => [...prev, userMsg]);

    // 3. Prepare assistant placeholder message
    const assistantMsgIndex = messages.length + 1;
    const initialAssistantMsg = {
      role: 'assistant',
      content: '',
      intent: null,
      confidence: null,
      rewriteInfo: null,
      citations: [],
      explainabilityData: {
        query: queryText,
        userRole: userRole,
        security: null,
        rewrite: null,
        intent: null,
        retrieval: null,
        rerank: null,
        metrics: null
      }
    };

    setMessages(prev => [...prev, initialAssistantMsg]);
    setIsStreaming(true);

    try {
      const response = await fetch('/api/documents/rag/stream/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: queryText,
          conversation_history: historyPayload,
          user_role: userRole
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const rawJson = line.slice(6).trim();
            if (!rawJson) continue;

            try {
              const event = JSON.parse(rawJson);
              const { type, payload } = event;

              setMessages(prev => {
                if (prev.length === 0) return prev;
                const next = [...prev];
                const lastIdx = next.length - 1;
                const target = { ...next[lastIdx] };
                const explain = { ...target.explainabilityData };

                if (type === 'security') {
                  explain.security = payload;
                } else if (type === 'rewrite') {
                  target.rewriteInfo = payload;
                  explain.rewrite = payload;
                } else if (type === 'intent') {
                  target.intent = payload.intent;
                  target.confidence = payload.confidence;
                  explain.intent = payload;
                } else if (type === 'retrieval') {
                  explain.retrieval = payload;
                } else if (type === 'rerank') {
                  explain.rerank = payload;
                } else if (type === 'token') {
                  target.content += payload.delta || '';
                } else if (type === 'metrics') {
                  explain.metrics = payload;
                  target.citations = payload.citations || [];
                  if (payload.latency_breakdown?.total_ms) {
                    setLatestLatency(payload.latency_breakdown.total_ms);
                  }
                } else if (type === 'done') {
                  explain.answer = target.content;
                }

                target.explainabilityData = explain;
                next[lastIdx] = target;
                return next;
              });
            } catch (jsonErr) {
              console.error('Error parsing SSE event', jsonErr, rawJson);
            }
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const next = [...prev];
        const target = { ...next[assistantMsgIndex] };
        target.content = `❌ Error streaming response: ${err.message}`;
        next[assistantMsgIndex] = target;
        return next;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleInspectMessage = (msg) => {
    if (msg.explainabilityData) {
      setExplainModalData({
        ...msg.explainabilityData,
        answer: msg.content
      });
    }
  };

  return (
    <div className="app-container">
      <Navbar 
        latency={latestLatency} 
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        userRole={userRole}
        setUserRole={setUserRole}
      />

      {activeTab === 'chat' ? (
        <div className="main-layout">
          <Sidebar 
            documents={documents}
            loadingDocs={loadingDocs}
            uploading={uploading}
            activeIngestJob={activeIngestJob}
            userRole={userRole}
            onUploadFile={handleUploadFile}
            onUpdateDocRole={handleUpdateDocumentRole}
            onRefreshDocs={fetchDocuments}
            onSelectPrompt={(q) => handleSendMessage(q)}
            onOpenCompare={(docA, docB) => setDiffModalDocs({ docA, docB })}
          />

          <ChatArea 
            messages={messages}
            inputValue={inputValue}
            setInputValue={setInputValue}
            onSendMessage={handleSendMessage}
            isStreaming={isStreaming}
            onInspectMessage={handleInspectMessage}
            onOpenCitation={(c) => setSelectedCitation(c)}
          />
        </div>
      ) : (
        <BenchmarkDashboard />
      )}

      {explainModalData && (
        <ExplainabilityModal 
          data={explainModalData} 
          onClose={() => setExplainModalData(null)} 
        />
      )}

      {selectedCitation && (
        <CitationModal 
          citation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
        />
      )}

      {diffModalDocs && (
        <VersionDiffModal 
          docA={diffModalDocs.docA}
          docB={diffModalDocs.docB}
          onClose={() => setDiffModalDocs(null)}
        />
      )}
    </div>
  );
}

