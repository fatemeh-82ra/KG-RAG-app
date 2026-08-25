import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'

const STATUS_LABELS = {
  no_docs: { text: 'No documents', cls: 'badge gray' },
  processing: { text: 'Indexing…', cls: 'badge amber' },
  ready: { text: 'Ready', cls: 'badge green' },
  error: { text: 'Error', cls: 'badge red' },
}

function Sidebar({ conversations, activeId, status, docs, onSelect, onNew, onDelete, onUpload }) {
  const fileRef = useRef(null)

  return (
    <aside className="sidebar">
      <button className="btn primary full" onClick={onNew}>+ New conversation</button>

      <div className="upload-box">
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt"
          hidden
          onChange={(e) => { onUpload(Array.from(e.target.files)); e.target.value = '' }}
        />
        <button
          className="btn full"
          disabled={!activeId || status?.status === 'processing'}
          onClick={() => fileRef.current?.click()}
        >
          📄 Upload documents
        </button>
        {activeId && (
          <>
            <div className="doc-status">
              <span className={STATUS_LABELS[status?.status || 'no_docs']?.cls}>
                {STATUS_LABELS[status?.status || 'no_docs']?.text}
              </span>
              {status?.detail && <span className="detail">{status.detail}</span>}
            </div>
            {docs.length > 0 && (
              <ul className="doc-list">
                {docs.map((d, i) => (
                  <li key={i} title={d.filename}>
                    {d.status === 'error' ? '⚠️ ' : d.status === 'ready' ? '✅ ' : '⏳ '}
                    {d.filename}
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>

      <div className="conv-list">
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`conv-item ${c.id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(c.id)}
          >
            <span className="conv-title">{c.title}</span>
            <button
              className="conv-delete"
              title="Delete"
              onClick={(e) => { e.stopPropagation(); if (confirm(`Delete "${c.title}"?`)) onDelete(c.id) }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </aside>
  )
}

function Message({ role, content, meta }) {
  return (
    <div className={`msg ${role}`}>
      <div className="bubble">
        {content}
        {meta && <div className="meta">{meta}</div>}
      </div>
    </div>
  )
}

export default function App() {
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [messages, setMessages] = useState([])
  const [status, setStatus] = useState(null)
  const [docs, setDocs] = useState([])
  const [models, setModels] = useState({ chat: [], embedding: [] })
  const [modelChoice, setModelChoice] = useState('')
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)

  const refreshConversations = useCallback(() => {
    api.fetchConversations().then(setConversations).catch(() => {})
  }, [])

  useEffect(() => { refreshConversations() ; api.fetchModels().then(setModels).catch(() => {}) }, [refreshConversations])

  // Poll ingestion status while processing
  useEffect(() => {
    if (!activeId) return
    let timer
    const poll = async () => {
      try {
        const st = await api.fetchStatus(activeId)
        setStatus(st)
        setDocs(await api.fetchDocuments(activeId))
        timer = setTimeout(poll, st.status === 'processing' ? 2000 : 8000)
      } catch { /* ignore */ }
    }
    poll()
    return () => clearTimeout(timer)
  }, [activeId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const selectConversation = async (id) => {
    setActiveId(id)
    setError('')
    try {
      setMessages(await api.fetchMessages(id))
    } catch { setMessages([]) }
  }

  const newConversation = async () => {
    const title = prompt('Conversation name:', `Chat ${conversations.length + 1}`)
    if (title === null) return
    const conv = await api.createConversation(title || 'New conversation')
    refreshConversations()
    selectConversation(conv.id)
  }

  const removeConversation = async (id) => {
    await api.deleteConversation(id)
    refreshConversations()
    if (id === activeId) { setActiveId(null); setMessages([]); setStatus(null); setDocs([]) }
  }

  const upload = async (files) => {
    if (!files.length || !activeId) return
    try {
      await api.uploadDocuments(activeId, files)
      setStatus({ status: 'processing', detail: 'Uploading…' })
    } catch (e) { setError(e.message) }
  }

  const ask = async () => {
    const q = input.trim()
    if (!q || !activeId || busy) return
    setInput('')
    setBusy(true)
    setError('')
    setMessages((m) => [...m, { role: 'user', content: q }])
    try {
      const [provider, model] = modelChoice ? modelChoice.split('|') : [null, null]
      const res = await api.sendQuestion(activeId, q, provider, model)
      setMessages((m) => [...m, {
        role: 'assistant',
        content: res.answer,
        meta: `${res.graph_facts} graph facts · ${res.chunks_used} chunks`,
      }])
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  const chatProviders = models.chat.filter((p) => p.available)

  return (
    <div className="layout">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        status={status}
        docs={docs}
        onSelect={selectConversation}
        onNew={newConversation}
        onDelete={removeConversation}
        onUpload={upload}
      />

      <main className="chat">
        {!activeId ? (
          <div className="placeholder">
            <h2>Hybrid Knowledge-Graph RAG</h2>
            <p>Create a conversation, upload PDF/DOCX/TXT documents, then ask questions.<br />
              Answers come from the Neo4j knowledge graph + ChromaDB semantic search.</p>
          </div>
        ) : (
          <>
            <div className="toolbar">
              <select value={modelChoice} onChange={(e) => setModelChoice(e.target.value)}>
                <option value="">Auto-select LLM</option>
                {chatProviders.map((p) =>
                  p.models.map((m) => (
                    <option key={`${p.provider}|${m}`} value={`${p.provider}|${m}`}>
                      {p.label} — {m}
                    </option>
                  )),
                )}
              </select>
            </div>

            <div className="messages">
              {messages.length === 0 && (
                <div className="hint">Ask a question about your uploaded documents (English or فارسی).</div>
              )}
              {messages.map((m, i) => <Message key={i} {...m} />)}
              {busy && <div className="typing">Thinking…</div>}
              <div ref={bottomRef} />
            </div>

            {error && <div className="error">{error}</div>}

            <div className="composer">
              <textarea
                rows={1}
                placeholder="Type your question…"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask() }
                }}
              />
              <button className="btn primary" disabled={busy || !input.trim()} onClick={ask}>
                Send
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
