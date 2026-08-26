// In dev, Vite proxies /api -> localhost:8000 (see vite.config.js).
// In production, set VITE_API_URL to your backend URL, e.g. https://my-app.onrender.com/api
const BASE = import.meta.env.VITE_API_URL || '/api'

const TOKEN_KEY = 'kg_rag_auth_token'

export const getAuthToken = () => localStorage.getItem(TOKEN_KEY)
export const setAuthToken = (t) => localStorage.setItem(TOKEN_KEY, t)
export const clearAuthToken = () => localStorage.removeItem(TOKEN_KEY)

function authHeaders(extra = {}) {
  const t = getAuthToken()
  return { ...(t ? { 'X-Auth-Token': t } : {}), ...extra }
}

export async function login(username, password) {
  const res = await fetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    let detail = 'Login failed'
    try { detail = (await res.json()).detail || detail } catch { /* ignore */ }
    throw new Error(detail)
  }
  const { token, username: u } = await res.json()
  setAuthToken(token)
  localStorage.setItem('kg_rag_username', u)
  return u
}

export async function signup(username, password, displayName) {
  const res = await fetch(`${BASE}/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, display_name: displayName || username }),
  })
  if (!res.ok) {
    let detail = 'Signup failed'
    try { detail = (await res.json()).detail || detail } catch { /* ignore */ }
    throw new Error(detail)
  }
  const { token, username: u } = await res.json()
  setAuthToken(token)
  localStorage.setItem('kg_rag_username', u)
  return u
}

export const getUsername = () => localStorage.getItem('kg_rag_username') || ''

export function logout() {
  clearAuthToken()
  localStorage.removeItem('kg_rag_username')
}

async function handle(res) {
  if (res.status === 401) {
    clearAuthToken()
    window.dispatchEvent(new Event('kg-rag-unauthorized'))
    throw new Error('Session expired — please log in again.')
  }
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
  fetch(`${BASE}/models`, { headers: authHeaders() }).then(handle)

export const fetchConversations = () =>
  fetch(`${BASE}/conversations`, { headers: authHeaders() }).then(handle)

export const createConversation = (title, systemPrompt = '') =>
  fetch(`${BASE}/conversations`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title, system_prompt: systemPrompt }),
  }).then(handle)

export const updateConversation = (id, { title, systemPrompt }) =>
  fetch(`${BASE}/conversations/${id}`, {
    method: 'PATCH',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title, system_prompt: systemPrompt }),
  }).then(handle)

export const deleteConversation = (id) =>
  fetch(`${BASE}/conversations/${id}`, {
    method: 'DELETE', headers: authHeaders(),
  }).then(handle)

export const uploadDocuments = (id, files) => {
  const fd = new FormData()
  files.forEach((f) => fd.append('files', f))
  return fetch(`${BASE}/conversations/${id}/documents`, {
    method: 'POST',
    headers: authHeaders(),
    body: fd,
  }).then(handle)
}

export const fetchStatus = (id) =>
  fetch(`${BASE}/conversations/${id}/status`, { headers: authHeaders() }).then(handle)

export const fetchDocuments = (id) =>
  fetch(`${BASE}/conversations/${id}/documents`, { headers: authHeaders() }).then(handle)

export const fetchMessages = (id) =>
  fetch(`${BASE}/conversations/${id}/messages`, { headers: authHeaders() }).then(handle)

export const sendQuestion = (id, question, chatProvider, chatModel) =>
  fetch(`${BASE}/conversations/${id}/chat`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      question,
      chat_provider: chatProvider || null,
      chat_model: chatModel || null,
    }),
  }).then(handle)

export const chatStreamURL = () => `${BASE}`

export const fetchProviders = () =>
  fetch(`${BASE}/providers`, { headers: authHeaders() }).then(handle)

export const addProvider = (p) =>
  fetch(`${BASE}/providers`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(p),
  }).then(handle)

export const deleteProvider = (id) =>
  fetch(`${BASE}/providers/${id}`, { method: 'DELETE', headers: authHeaders() }).then(handle)
