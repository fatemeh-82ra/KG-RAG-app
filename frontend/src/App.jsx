import React, { useCallback, useEffect, useRef, useState } from 'react'
import html2pdf from 'html2pdf.js'
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
  const [memory, setMemory] = useState(initial?.memory_turns ?? 5)
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
      <label className="field">
        <span>Chat memory — how many of the last Q/A turns are sent with every question (0 = off). The bot can answer follow-ups from these turns.</span>
        <select value={memory} onChange={(e) => setMemory(Number(e.target.value))}>
          {[0, 1, 2, 3, 5, 8, 10, 15, 20].map((n) => (
            <option key={n} value={n}>
              {n === 0 ? 'No memory' : `Last ${n} turn${n > 1 ? 's' : ''}`}
            </option>
          ))}
        </select>
      </label>
      <div className="modal-actions">
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary"
                disabled={!title.trim()}
                onClick={() => onSave(title.trim(), sys, memory)}>
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

function Message({ role, content, meta, id, feedback, onFeedback, onRepeat }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1300)
    } catch { /* clipboard unavailable */ }
  }
  return (
    <div className={`msg ${role}`}>
      <div className="bubble">
        {content}
        {meta && <div className="meta">{meta}</div>}
        {content && (
          <div className="msg-actions">
            {role === 'assistant' && id && onFeedback && (
              <>
                <button
                  className={`msg-btn ${feedback === 'like' ? 'active like' : ''}`}
                  title="Good answer"
                  onClick={() => onFeedback(id, feedback === 'like' ? '' : 'like')}
                >👍</button>
                <button
                  className={`msg-btn ${feedback === 'dislike' ? 'active dislike' : ''}`}
                  title="Bad answer — excluded from chat memory"
                  onClick={() => onFeedback(id, feedback === 'dislike' ? '' : 'dislike')}
                >👎</button>
              </>
            )}
            <button className="msg-btn" title={copied ? 'Copied!' : 'Copy'} onClick={copy}>
              {copied ? '✅' : '📋'}
            </button>
            {role === 'user' && onRepeat && (
              <button className="msg-btn" title="Ask this question again"
                      onClick={() => onRepeat(content)}>
                🔁
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ShareModal({ url, onRevoke, onClose }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard unavailable */ }
  }
  return (
    <Modal title="Share conversation" onClose={onClose}>
      <p className="modal-hint">
        Anyone with this link can view a read-only copy of this chat — no login needed.
        Use &quot;Revoke link&quot; to disable it again at any time.
      </p>
      <div className="share-url">{url}</div>
      <div className="modal-actions">
        <button className="btn" onClick={onRevoke}>Revoke link</button>
        <button className="btn primary" onClick={copy}>{copied ? '✅ Copied!' : '📋 Copy link'}</button>
      </div>
    </Modal>
  )
}

// Public read-only view for people opening a share link (#/share/<token>)
function SharedChat({ token }) {
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    api.fetchShared(token).then(setData).catch((e) => setErr(e.message))
  }, [token])
  return (
    <div className="shared-wrap">
      <div className="shared-head">
        <h2>🔗 {data?.title || 'Shared chat'}</h2>
        <span>Read-only shared conversation — KG-RAG</span>
      </div>
      <div className="shared-body">
        {err && <div className="error">{err}</div>}
        {!err && !data && <div className="hint">Loading…</div>}
        {data && data.messages.length === 0 && <div className="hint">This conversation is empty.</div>}
        {data?.messages.map((m, i) => <Message key={i} {...m} />)}
      </div>
    </div>
  )
}

export default function App() {
  // public shared-chat route: #/share/<token>
  const [shareToken] = useState(() => {
    const m = window.location.hash.match(/^#\/share\/([A-Za-z0-9_-]+)/)
    return m ? m[1] : null
  })
  if (shareToken) return <SharedChat token={shareToken} />
  return <ChatApp />
}

function ChatApp() {
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
  const [memoryTurns, setMemoryTurns] = useState(5) // chat memory of active conversation
  const [exportFormat, setExportFormat] = useState('pdf') // chat export format: pdf | txt
  const [exporting, setExporting] = useState(false)     // PDF export in progress
  const [authed, setAuthed] = useState(!!api.getAuthToken())
  const bottomRef = useRef(null)
  const messagesRef = useRef(null)

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
    const conv = conversations.find((c) => c.id === id)
    setMemoryTurns(conv?.memory_turns ?? 5)
    try {
      setMessages(await api.fetchMessages(id))
    } catch { setMessages([]) }
  }

  const newConversation = () => setModal({ mode: 'new' })

  const editConversation = async (id) => {
    const conv = conversations.find((c) => c.id === id)
    if (conv) setModal({ mode: 'edit', conv })
  }

  const saveConversation = async (title, systemPrompt, memory) => {
    try {
      if (modal?.mode === 'edit' && modal.conv) {
        await api.updateConversation(modal.conv.id, { title, systemPrompt, memoryTurns: memory })
        setMemoryTurns(memory)
      } else {
        const conv = await api.createConversation(title, systemPrompt, memory)
        refreshConversations()
        setModal(null)
        selectConversation(conv.id)
        return
      }
      refreshConversations()
    } catch (e) { setError(e.message) }
    setModal(null)
  }

  // Quick memory change from the toolbar (persists immediately)
  const changeMemory = async (n) => {
    setMemoryTurns(n)
    if (!activeId) return
    try {
      await api.updateConversation(activeId, { memoryTurns: n })
      refreshConversations()
    } catch (e) { setError(e.message) }
  }

  const openProviders = async () => {
    setModal({ mode: 'providers' })
    try { setProviders(await api.fetchProviders()) } catch { setProviders([]) }
  }

  // Share: create (or reuse) a public read-only link for the active conversation
  const openShare = async () => {
    if (!activeId) return
    try {
      const { share_id } = await api.shareConversation(activeId)
      const url = `${window.location.origin}${window.location.pathname}#/share/${share_id}`
      setModal({ mode: 'share', url, cid: activeId })
    } catch (e) { setError(e.message) }
  }

  const revokeShare = async () => {
    try {
      await api.revokeShare(modal.cid)
      setModal(null)
    } catch (e) { setError(e.message) }
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

  // Export the active conversation: PDF (same visual style) or plain text
  const exportChat = async () => {
    if (!activeId || messages.length === 0 || exporting) return
    const conv = conversations.find((c) => c.id === activeId)
    const title = conv?.title || 'chat'
    const safe = (title.replace(/[\\/:*?"<>|]+/g, '_').trim() || 'chat').slice(0, 60)
    const stamp = new Date().toISOString().slice(0, 10)

    if (exportFormat === 'pdf') {
      const el = messagesRef.current
      if (!el) return
      setExporting(true)
      el.classList.add('exporting')
      // let the container grow to its full content height so html2canvas
      // captures the whole chat, not just the scrolled viewport
      const prev = { h: el.style.height, mh: el.style.maxHeight, ov: el.style.overflow, fx: el.style.flex }
      el.style.height = `${el.scrollHeight}px`
      el.style.maxHeight = 'none'
      el.style.overflow = 'visible'
      el.style.flex = '0 0 auto'
      try {
        await html2pdf().set({
          margin: [8, 8, 8, 8],
          filename: `${safe}_${stamp}.pdf`,
          image: { type: 'jpeg', quality: 0.95 },
          html2canvas: { scale: 2, backgroundColor: '#0f1117', useCORS: true },
          jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
          pagebreak: { mode: ['css', 'legacy'] },
        }).from(el).save()
      } catch (e) {
        setError(`PDF export failed: ${e.message}`)
      } finally {
        el.style.height = prev.h
        el.style.maxHeight = prev.mh
        el.style.overflow = prev.ov
        el.style.flex = prev.fx
        el.classList.remove('exporting')
        setExporting(false)
      }
      return
    }

    // plain text
    const now = new Date()
    const content = `${title}\nExported: ${now.toLocaleString()}\n${'='.repeat(40)}\n\n` +
      messages.map((m) => `${m.role === 'user' ? 'User' : 'Bot'}: ${m.content}`).join('\n\n')
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${safe}_${stamp}.txt`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
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

  const ask = async (questionArg) => {
    const q = (questionArg ?? input).trim()
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
      // Re-sync so fresh messages get their DB ids (needed for 👍/👎 feedback)
      try { setMessages(await api.fetchMessages(activeId)) } catch { /* keep streamed */ }
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

  // Rate an assistant answer ('like' / 'dislike' / ''). Disliked answers leave chat memory.
  const handleFeedback = async (messageId, fb) => {
    setMessages((msgs) => msgs.map((m) => (m.id === messageId ? { ...m, feedback: fb } : m)))
    try { await api.setMessageFeedback(activeId, messageId, fb) } catch (e) { setError(e.message) }
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
              <select value={memoryTurns} onChange={(e) => changeMemory(Number(e.target.value))}
                      title="Chat memory — how many recent Q/A turns are sent with each question">
                {[0, 1, 2, 3, 5, 8, 10, 15, 20].map((n) => (
                  <option key={n} value={n}>
                    {n === 0 ? '🧠 Memory: off' : `🧠 Memory: last ${n}`}
                  </option>
                ))}
              </select>
              <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}
                      title="File format for the chat export">
                <option value="pdf">PDF (.pdf)</option>
                <option value="txt">Plain text (.txt)</option>
              </select>
              <button className="btn" onClick={exportChat}
                      disabled={!activeId || messages.length === 0 || exporting}
                      title="Download the whole conversation as a file">
                {exporting ? '⏳ …' : '⬇ Export'}
              </button>
              <button className="btn" onClick={openShare}
                      disabled={!activeId || messages.length === 0}
                      title="Create a public read-only link for this chat">
                🔗 Share
              </button>
              <button className="btn" onClick={openProviders} title="Manage custom LLM providers">
                ⚙ Providers
              </button>
            </div>

            <div className={`messages${exporting ? ' exporting' : ''}`} ref={messagesRef}>
              {messages.length === 0 && (
                <div className="hint">Ask a question about your uploaded documents (English or فارسی).</div>
              )}
              {messages.map((m, i) => (
                <Message key={i} {...m}
                         onFeedback={handleFeedback}
                         onRepeat={(text) => ask(text)} />
              ))}
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
              <button className="btn primary" disabled={busy || !input.trim()} onClick={() => ask()}>
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
      {modal?.mode === 'share' && (
        <ShareModal
          url={modal.url}
          onRevoke={revokeShare}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  )
}
