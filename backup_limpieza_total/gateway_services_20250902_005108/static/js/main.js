'use strict';

// Read global config from HTML
const CFG = window.AgenteIA?.config || {};
const API_BASE = CFG.API_BASE || window.location.origin;
const WS_URL = CFG.WS_URL || ((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws');
const USE_MOCKS = (localStorage.getItem('USE_MOCKS') !== '0');

async function api(path, opts={}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: opts.method || 'GET',
    headers: {'Content-Type':'application/json', ...(opts.headers||{})},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const msg = await res.text().catch(()=>res.statusText);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return res.json();
}

// WebSocket indicator
(function connectWS(){
  try {
    const el = document.getElementById('ws-indicator');
    const ws = new WebSocket(WS_URL);
    ws.onopen = ()=>{ if(el){ el.title='WS Conectado'; el.classList.add('text-green'); } };
    ws.onclose = ()=>{ if(el){ el.title='WS Desconectado'; el.classList.remove('text-green'); } };
    ws.onerror = ()=>{ if(el){ el.title='WS Error'; el.classList.add('text-red'); } };
  } catch(e){ console.warn('WS error', e); }
})();

// Simple alert utility
function showMsg(text, type='info') {
  const box = document.getElementById('alert-box') || document.body;
  const div = document.createElement('div');
  div.className = `alert alert-${type}`;
  div.innerText = text;
  box.appendChild(div);
  setTimeout(()=>div.remove(), 4000);
}

// Global namespace
window.OyP = { api, showMsg, API_BASE, USE_MOCKS };
console.log('✅ main.js cargado', {API_BASE, USE_MOCKS});
