// In dev, Vite proxies /api -> localhost:8000 (see vite.config.js).
// In production, set VITE_API_URL to your backend URL, e.g. https://my-app.onrender.com/api
const BASE = import.meta.env.VITE_API_URL || '/api'

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch { /* ignore */ }
    throw new Error(detail)
  }
  return res.json()
}

export const fetchModels = () =>
  fetch(`${BASE}/models`).then(handle)

export const fetchConversations = () =>
  fetch(`${BASE}/conversations`).then(handle)

export const createConversation = (title, systemPrompt = '') =>
  fetch(`${BASE}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, system_prompt: systemPrompt }),
  }).then(handle)

export const updateConversation = (id, { title, systemPrompt }) =>
  fetch(`${BASE}/conversations/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, system_prompt: systemPrompt }),
  }).then(handle)

export const deleteConversation = (id) =>
  fetch(`${BASE}/conversations/${id}`, { method: 'DELETE' }).then(handle)

export const uploadDocuments = (id, files) => {
  const fd = new FormData()
  files.forEach((f) => fd.append('files', f))
  return fetch(`${BASE}/conversations/${id}/documents`, {
    method: 'POST',
    body: fd,
  }).then(handle)
}

export const fetchStatus = (id) =>
  fetch(`${BASE}/conversations/${id}/status`).then(handle)

export const fetchDocuments = (id) =>
  fetch(`${BASE}/conversations/${id}/documents`).then(handle)

export const fetchMessages = (id) =>
  fetch(`${BASE}/conversations/${id}/messages`).then(handle)

export const sendQuestion = (id, question, chatProvider, chatModel) =>
  fetch(`${BASE}/conversations/${id}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      chat_provider: chatProvider || null,
      chat_model: chatModel || null,
    }),
  }).then(handle)

export const fetchProviders = () =>
  fetch(`${BASE}/providers`).then(handle)

export const addProvider = (p) =>
  fetch(`${BASE}/providers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(p),
  }).then(handle)

export const deleteProvider = (id) =>
  fetch(`${BASE}/providers/${id}`, { method: 'DELETE' }).then(handle)

export const fetchConversation = (id) =>
  fetch(`${BASE}/conversations`).then((r) => r.json()).then((list) =>
    list.find((c) => c.id === id))
