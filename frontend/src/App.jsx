import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'

const STATUS_LABELS = {
  no_docs: { text: 'No documents', cls: 'badge gray' },
  processing: { text: 'Indexing…', cls: 'badge amber' },
  ready: { text: 'Ready', cls: 'badge green' },
  error: { text: 'Error', cls: 'badge red' },
}

function Sidebar({ conversations, activeId, status, docs, onSelect, onNew, onEdit, onDelete, onUpload }) {
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
            <span className="conv-actions">
              <button
                className="conv-delete"
                title="Edit name / bot instructions"
                onClick={(e) => { e.stopPropagation(); onEdit(c.id) }}
              >
                ✎
              </button>
              <button
                className="conv-delete"
                title="Delete"
                onClick={(e) => { e.stopPropagation(); if (confirm(`Delete "${c.title}"?`)) onDelete(c.id) }}
              >
                ×
              </button>
            </span>
          </div>
        ))}
      </div>
    </aside>
  )
}

function Modal({ title, children, onClose }) {
  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal">
        <div className="modal-head">
          <h3>{title}</h3>
          <button className="conv-delete" onClick={onClose}>×</button>
        </div>
        {children}
      </div>
    </div>
  )
}

function ConversationModal({ initial, onSave, onClose }) {
  const [title, setTitle] = useState(initial?.title || '')
  const [sys, setSys] = useState(initial?.system_prompt || '')
  return (
    <Modal title={initial?.id ? 'Edit conversation' : 'New conversation'} onClose={onClose}>
      <label className="field">
        <span>Name</span>
        <input value={title} onChange={(e) => setTitle(e.target.value)}
               placeholder="e.g. Bachelor thesis docs" autoFocus />
      </label>
      <label className="field">
        <span>Bot instructions (optional — persona, tone, what to say when info is missing…)</span>
        <textarea rows={5} value={sys} onChange={(e) => setSys(e.target.value)}
                  placeholder={'Example:\nYou are the secretary of the university council. Answer formally in Persian. If the answer is not in the documents, politely say you cannot find it and suggest contacting the council office.'} />
      </label>
      <div className="modal-actions">
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary"
                disabled={!title.trim()}
                onClick={() => onSave(title.trim(), sys)}>
          {initial?.id ? 'Save' : 'Create'}
        </button>
      </div>
    </Modal>
  )
}

function ProvidersModal({ providers, onAdd, onDelete, onClose }) {
  const [form, setForm] = useState({ name: '', base_url: '', api_key: '', chat_model: '', embedding_model: '' })
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })
  return (
    <Modal title="LLM providers" onClose={onClose}>
      <p className="modal-hint">
        Add any OpenAI-compatible API (e.g. gapgpt, OpenRouter, a self-hosted vLLM…).
        Custom providers are tried after the built-in ones and also appear in the model dropdown.
      </p>

      <div className="provider-list">
        {providers.length === 0 && <div className="hint">No custom providers yet.</div>}
        {providers.map((p) => (
          <div key={p.id} className="provider-item">
            <div>
              <b>{p.name}</b>
              <div className="provider-meta">
                {p.chat_model && <>💬 {p.chat_model} </>}
                {p.embedding_model && <>· 🔢 {p.embedding_model}</>}
              </div>
              <div className="provider-meta url">{p.base_url}</div>
            </div>
            <button className="conv-delete" title="Remove" onClick={() => onDelete(p.id)}>×</button>
          </div>
        ))}
      </div>

      <div className="provider-form">
        <label className="field"><span>Name *</span>
          <input value={form.name} onChange={set('name')} placeholder="gapgpt" /></label>
        <label className="field"><span>Base URL *</span>
          <input value={form.base_url} onChange={set('base_url')} placeholder="https://api.example.com/v1" /></label>
        <label className="field"><span>API key</span>
          <input type="password" value={form.api_key} onChange={set('api_key')} placeholder="sk-…" /></label>
        <label className="field"><span>Chat model</span>
          <input value={form.chat_model} onChange={set('chat_model')} placeholder="gpt-4o-mini" /></label>
        <label className="field"><span>Embedding model (optional)</span>
          <input value={form.embedding_model} onChange={set('embedding_model')} placeholder="text-embedding-3-small" /></label>
        <button className="btn primary full"
                disabled={!form.name.trim() || !form.base_url.trim()}
                onClick={() => { onAdd(form); setForm({ name: '', base_url: '', api_key: '', chat_model: '', embedding_model: '' }) }}>
          + Add provider
        </button>
      </div>
    </Modal>
  )
}

function LoginScreen({ onLogin }) {
  const [mode, setMode] = useState('login')          // 'login' | 'signup'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    if (!username || !password || busy) return
    setBusy(true); setErr('')
    try {
      if (mode === 'signup') await api.signup(username, password)
      else await api.login(username, password)
      onLogin()
    } catch (ex) { setErr(ex.message) }
    finally { setBusy(false) }
  }

  return (
    <div className="login-wrap">
      <form className="login-box" onSubmit={submit}>
        <h2>🔐 KG-RAG</h2>
        <p>Hybrid Knowledge-Graph RAG</p>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoFocus
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {err && <div className="login-err">{err}</div>}
        <button className="btn primary full" disabled={busy || !username || !password}>
          {busy ? 'Please wait…' : (mode === 'signup' ? 'Create account' : 'Log in')}
        </button>
        <div className="login-switch">
          {mode === 'login' ? (
            <>Don&apos;t have an account?{' '}
              <button type="button" className="linklike" onClick={() => { setMode('signup'); setErr('') }}>
                Sign up
              </button>
            </>
          ) : (
            <>Already have an account?{' '}
              <button type="button" className="linklike" onClick={() => { setMode('login'); setErr('') }}>
                Log in
              </button>
            </>
          )}
        </div>
      </form>
    </div>
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
  const [graphModelMode, setGraphModelMode] = useState('auto')  // 'auto' | 'dropdown'
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [modal, setModal] = useState(null)          // null | {mode:'new'} | {mode:'edit',conv} | {mode:'providers'}
  const [providers, setProviders] = useState([])
  const [authed, setAuthed] = useState(!!api.getAuthToken())
  const bottomRef = useRef(null)

  const refreshConversations = useCallback(() => {
    api.fetchConversations().then(setConversations).catch(() => {})
  }, [])

  useEffect(() => {
    const onUnauthorized = () => setAuthed(false)
    window.addEventListener('kg-rag-unauthorized', onUnauthorized)
    return () => window.removeEventListener('kg-rag-unauthorized', onUnauthorized)
  }, [])

  useEffect(() => {
    if (authed) {
      refreshConversations()
      api.fetchModels().then(setModels).catch(() => {})
    }
  }, [authed, refreshConversations])

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

  const newConversation = () => setModal({ mode: 'new' })

  const editConversation = async (id) => {
    const conv = conversations.find((c) => c.id === id)
    if (conv) setModal({ mode: 'edit', conv })
  }

  const saveConversation = async (title, systemPrompt) => {
    try {
      if (modal?.mode === 'edit' && modal.conv) {
        await api.updateConversation(modal.conv.id, { title, systemPrompt })
      } else {
        const conv = await api.createConversation(title, systemPrompt)
        refreshConversations()
        setModal(null)
        selectConversation(conv.id)
        return
      }
      refreshConversations()
    } catch (e) { setError(e.message) }
    setModal(null)
  }

  const openProviders = async () => {
    setModal({ mode: 'providers' })
    try { setProviders(await api.fetchProviders()) } catch { setProviders([]) }
  }

  const addProvider = async (p) => {
    try {
      await api.addProvider(p)
      setProviders(await api.fetchProviders())
      api.fetchModels().then(setModels).catch(() => {})
    } catch (e) { setError(e.message) }
  }

  const removeProvider = async (id) => {
    try {
      await api.deleteProvider(id)
      setProviders(await api.fetchProviders())
      api.fetchModels().then(setModels).catch(() => {})
    } catch (e) { setError(e.message) }
  }

  const removeConversation = async (id) => {
    await api.deleteConversation(id)
    refreshConversations()
    if (id === activeId) { setActiveId(null); setMessages([]); setStatus(null); setDocs([]) }
  }

  const upload = async (files) => {
    if (!files.length || !activeId) return
    try {
      // graphModelMode: 'auto' -> default chain; 'dropdown' -> use selected chat model
      await api.uploadDocuments(activeId, files,
        graphModelMode === 'dropdown' ? modelChoice : '')
      setStatus({ status: 'processing', detail: 'Uploading…' })
    } catch (e) { setError(e.message) }
  }

  const ask = async () => {
    const q = input.trim()
    if (!q || !activeId || busy) return
    setInput('')
    setBusy(true)
    setError('')
    setMessages((m) => [...m, { role: 'user', content: q }, { role: 'assistant', content: '' }])
    try {
      const [provider, model] = modelChoice ? modelChoice.split('|') : [null, null]
      const res = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/conversations/${activeId}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(api.getAuthToken() ? { 'X-Auth-Token': api.getAuthToken() } : {}),
        },
        body: JSON.stringify({ question: q, chat_provider: provider || null, chat_model: model || null }),
      })
      if (!res.ok) {
        let detail = res.statusText
        try { detail = (await res.json()).detail || detail } catch { /* ignore */ }
        throw new Error(detail)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let meta = null
      let streamed = ''

      const applyMeta = (m) => {
        if (!m) return
        meta = m
        setMessages((msgs) => {
          const copy = [...msgs]
          copy[copy.length - 1] = { role: 'assistant', content: streamed || '…', meta: `${m.graph_facts} graph facts · ${m.chunks_used} chunks` }
          return copy
        })
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''
        for (const ev of events) {
          const line = ev.trim()
          if (!line.startsWith('data:')) continue
          let payload
          try { payload = JSON.parse(line.slice(5).trim()) } catch { continue }
          if (payload.type === 'meta') applyMeta(payload)
          else if (payload.type === 'token') {
            streamed += payload.text
            setMessages((msgs) => {
              const copy = [...msgs]
              copy[copy.length - 1] = {
                role: 'assistant',
                content: streamed,
                meta: meta ? `${meta.graph_facts} graph facts · ${meta.chunks_used} chunks` : undefined,
              }
              return copy
            })
          } else if (payload.type === 'error') {
            throw new Error(payload.detail || 'Stream error')
          }
        }
      }
    } catch (e) {
      setError(e.message)
      setMessages((msgs) => {
        const copy = [...msgs]
        if (copy.length && copy[copy.length - 1].role === 'assistant' && !copy[copy.length - 1].content) {
          copy.pop()
        }
        return copy
      })
    }
    finally { setBusy(false) }
  }

  const chatProviders = models.chat.filter((p) => p.available)

  if (!authed) return <LoginScreen onLogin={() => setAuthed(true)} />

  return (
    <div className="layout">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        status={status}
        docs={docs}
        onSelect={selectConversation}
        onNew={newConversation}
        onEdit={editConversation}
        onDelete={removeConversation}
        onUpload={upload}
      />

      <main className="chat">
        <div className="app-bar">
          <span className="app-title">KG-RAG</span>
          <span className="spacer" />
          <span className="username-chip" title={api.getUsername()}>
            👤 {api.getUsername()}
          </span>
          <button className="btn" title="Log out"
                  onClick={() => { api.logout(); setAuthed(false) }}>
            Log out ⎋
          </button>
        </div>

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
              <select value={graphModelMode} onChange={(e) => setGraphModelMode(e.target.value)}
                      title="Which model builds the knowledge graph when you upload files">
                <option value="auto">Graph model: Auto (fallback chain)</option>
                <option value="dropdown">Graph model: Same as chat model</option>
              </select>
              <button className="btn" onClick={openProviders} title="Manage custom LLM providers">
                ⚙ Providers
              </button>
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

      {modal?.mode === 'new' && (
        <ConversationModal
          initial={{ title: `Chat ${conversations.length + 1}`, system_prompt: '' }}
          onSave={saveConversation}
          onClose={() => setModal(null)}
        />
      )}
      {modal?.mode === 'edit' && (
        <ConversationModal
          initial={modal.conv}
          onSave={saveConversation}
          onClose={() => setModal(null)}
        />
      )}
      {modal?.mode === 'providers' && (
        <ProvidersModal
          providers={providers}
          onAdd={addProvider}
          onDelete={removeProvider}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  )
}
