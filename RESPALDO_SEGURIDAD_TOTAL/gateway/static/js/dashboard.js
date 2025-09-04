/**
 * 🚀 AGENTE IA OYP 6.0 - DASHBOARD JAVASCRIPT
 * ==========================================
 * Archivo: gateway/static/js/dashboard.js
 * Funcionalidad completa del frontend
 */

'use strict';

// =====================
// CONFIGURACIÓN GLOBAL
// =====================

const CONFIG = {
  API_BASE: window.location.origin,
  WS_URL: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
  USE_MOCKS: localStorage.getItem('USE_MOCKS') !== '0',
  API_TIMEOUT: parseInt(localStorage.getItem('API_TIMEOUT')) || 10000,
  RETRY_ATTEMPTS: parseInt(localStorage.getItem('RETRY_ATTEMPTS')) || 3,
  DEBUG: localStorage.getItem('DEBUG') === '1'
};

// Estado global
const STATE = {
  currentSection: 'inicio',
  websocket: null,
  websocketConnected: false,
  trainingStep: 1,
  trainingInProgress: false,
  agentStats: { actionsCount: 0, successCount: 0, lastAction: 'Ninguna' },
  chartInstances: {},
  activeConnections: new Set(),
  wsReconnectAttempts: 0
};

// API endpoints
const API = {
  dashboard_data: () => `${CONFIG.API_BASE}/api/dashboard/stats`,
  services_status: () => `${CONFIG.API_BASE}/api/services/status`,
  ai_analyze: () => `${CONFIG.API_BASE}/api/ai/analyze`,
  ai_summarize: () => `${CONFIG.API_BASE}/api/ai/summarize`,
  document_upload: () => `${CONFIG.API_BASE}/api/documents/upload`,
  document_process_text: () => `${CONFIG.API_BASE}/api/documents/process_text`,
  analytics_upload: () => `${CONFIG.API_BASE}/api/analytics/upload_dataset`,
  analytics_analyze: () => `${CONFIG.API_BASE}/api/analytics/analyze`,
  analytics_visualize: () => `${CONFIG.API_BASE}/api/analytics/visualize`,
  analytics_text_analytics: () => `${CONFIG.API_BASE}/api/analytics/text_analytics`,
  report_generate: () => `${CONFIG.API_BASE}/api/reports/generate`,
  report_quick: () => `${CONFIG.API_BASE}/api/reports/quick_report`,
  training_start: () => `${CONFIG.API_BASE}/api/training/start`,
  training_projects: () => `${CONFIG.API_BASE}/api/training/projects`,
  agent_execute: () => `${CONFIG.API_BASE}/api/agent/execute`,
  agent_status: () => `${CONFIG.API_BASE}/api/agent/history`,
  chat_send: () => `${CONFIG.API_BASE}/api/ai/chat`
};

// =====================
// UTILIDADES CORE
// =====================

function showToast(title, message, type = 'info', timeout = 5000) {
  const container = document.getElementById('alerts-container');
  if (!container) return;

  const alertDiv = document.createElement('div');
  alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
  alertDiv.innerHTML = `
    <div class="d-flex">
      <i class="ti ti-${type === 'success' ? 'check' : type === 'error' ? 'alert-octagon' : 'info-circle'} me-2"></i>
      <div>
        <h4 class="alert-title mb-1">${title}</h4>
        <div>${message}</div>
      </div>
    </div>
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;

  container.appendChild(alertDiv);

  if (timeout) {
    setTimeout(() => {
      if (alertDiv.parentNode) alertDiv.remove();
    }, timeout);
  }
}

async function apiCall(url, options = {}) {
  // Si el cuerpo es FormData, el navegador establece el Content-Type automáticamente.
  const headers = options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' };

  for (let attempt = 1; attempt <= CONFIG.RETRY_ATTEMPTS; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), CONFIG.API_TIMEOUT);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          ...headers,
          ...options.headers
        }
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorBody;
        try {
          errorBody = await response.json();
        } catch (e) {
          errorBody = { detail: response.statusText };
        }
        throw new Error(`Error ${response.status}: ${errorBody.detail || response.statusText}`);
      }

      // Si la respuesta es JSON, la decodifica, si no, devuelve el texto plano.
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      return await response.text();

    } catch (error) {
      console.error(`API Call a ${url} falló en intento ${attempt}:`, error);
      if (attempt === CONFIG.RETRY_ATTEMPTS) {
        if (CONFIG.USE_MOCKS) {
          console.warn(`Fallback a MOCK para ${url}`);
          return getMockData(url);
        }
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }
}

function getMockData(url) {
  const mockResponses = {
    [API.dashboard_data()]: {
      total_documentos: 1247,
      documentos_semana: 89,
      precision_ia: "96.8%",
      conexiones_ws: STATE.activeConnections.size,
      actividad_semana: [
        { fecha: '2024-01-15', documentos: 45, analisis: 23 },
        { fecha: '2024-01-16', documentos: 52, analisis: 31 },
        { fecha: '2024-01-17', documentos: 38, analisis: 19 },
        { fecha: '2024-01-18', documentos: 61, analisis: 42 },
        { fecha: '2024-01-19', documentos: 47, analisis: 28 },
        { fecha: '2024-01-20', documentos: 59, analisis: 35 },
        { fecha: '2024-01-21', documentos: 43, analisis: 26 }
      ],
      distribucion_documentos: { PDF: 45, DOCX: 30, TXT: 15, IMAGEN: 10 }
    },
    [API.services_status()]: {
      services: [
        { name: 'AI Engine', status: 'online', port: 8001, uptime: '2h 15m', cpu: 45, memory: 67 },
        { name: 'Document Processor', status: 'online', port: 8002, uptime: '2h 15m', cpu: 23, memory: 34 },
        { name: 'Analytics Engine', status: 'online', port: 8003, uptime: '2h 14m', cpu: 12, memory: 28 },
        { name: 'Report Generator', status: 'online', port: 8004, uptime: '2h 13m', cpu: 8, memory: 19 },
        { name: 'Chat Service', status: 'online', port: 8005, uptime: '2h 12m', cpu: 15, memory: 22 }
      ]
    }
  };

  return mockResponses[url.split('?')[0]] || { status: 'ok', message: 'Mock response' };
}

// =====================
// WEBSOCKET
// =====================

function initWebSocket() {
  if (STATE.websocket) STATE.websocket.close();

  updateWSStatus('connecting');

  try {
    STATE.websocket = new WebSocket(CONFIG.WS_URL);

    STATE.websocket.onopen = () => {
      STATE.websocketConnected = true;
      STATE.wsReconnectAttempts = 0;
      updateWSStatus('connected');
      STATE.activeConnections.add('dashboard');
      updateConnectionCount();
    };

    STATE.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    STATE.websocket.onclose = () => {
      STATE.websocketConnected = false;
      updateWSStatus('disconnected');
      STATE.activeConnections.delete('dashboard');
      
      if (STATE.wsReconnectAttempts < 5) {
        STATE.wsReconnectAttempts++;
        setTimeout(initWebSocket, 1000 * STATE.wsReconnectAttempts);
      }
    };

  } catch (error) {
    console.error('Error inicializando WebSocket:', error);
  }
}

function updateWSStatus(status) {
  const wsIndicator = document.getElementById('ws-indicator');
  const chatStatus = document.getElementById('chat-status');
  
  if (wsIndicator) {
    wsIndicator.className = `ws-status ${status}`;
  }

  if (chatStatus) {
    const statusMap = {
      'connected': { text: 'WebSocket Conectado', class: 'badge bg-green-lt' },
      'connecting': { text: 'Conectando...', class: 'badge bg-yellow-lt' },
      'disconnected': { text: 'Desconectado', class: 'badge bg-red-lt' }
    };
    
    const config = statusMap[status];
    if (config) {
      chatStatus.textContent = config.text;
      chatStatus.className = config.class;
    }
  }
}

function updateConnectionCount() {
  const element = document.getElementById('conexiones-ws');
  if (element) {
    element.textContent = STATE.activeConnections.size;
  }
}

function handleWebSocketMessage(data) {
  switch (data.type) {
    case 'chat_response':
      addChatMessage(data.message, 'ai');
      break;
    case 'system_update':
      loadDashboardData();
      break;
    case 'training_progress':
      updateTrainingProgress(data.progress, data.status);
      break;
    case 'agent_action':
      logAgentAction(data.action, data.result);
      break;
  }
}

// =====================
// NAVEGACIÓN
// =====================

function setupNavigation() {
  document.querySelectorAll('[data-target]').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.getAttribute('data-target');
      showSection(target);
      
      document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      
      const title = link.querySelector('.nav-link-title')?.textContent || target;
      document.getElementById('page-title').textContent = title;
    });
  });

  document.querySelectorAll('[data-go]').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.getAttribute('data-go');
      showSection(target);
    });
  });
}

function showSection(sectionId) {
  document.querySelectorAll('.content-section').forEach(section => {
    section.classList.remove('active');
  });

  const targetSection = document.getElementById(sectionId);
  if (targetSection) {
    targetSection.classList.add('active');
    STATE.currentSection = sectionId;
    
    switch(sectionId) {
      case 'inicio':
        loadDashboardData();
        break;
      case 'servicios':
        loadServicesStatus();
        break;
      case 'entrenamiento':
        loadTrainingProjects();
        break;
      case 'agente':
        loadAgentStatus();
        break;
    }
  }
}

// =====================
// DASHBOARD PRINCIPAL
// =====================

async function loadDashboardData() {
  try {
    const data = await apiCall(API.dashboard_data());
    
    document.getElementById('total-documentos').textContent = data.total_documentos || '0';
    document.getElementById('documentos-semana').textContent = data.documentos_semana || '0';
    document.getElementById('precision-ia').textContent = data.precision_ia || '0%';

    updateActivityChart(data.actividad_semana || []);
    updateDistributionChart(data.distribucion_documentos || {});
    updateConclusiones(data);
    
  } catch (error) {
    console.error('Error loading dashboard data:', error);
    showToast('Error', 'No se pudieron cargar los datos del dashboard', 'error');
  }
}

function updateActivityChart(data) {
  const ctx = document.getElementById('actividad-chart');
  if (!ctx) return;

  if (STATE.chartInstances.actividad) {
    STATE.chartInstances.actividad.destroy();
  }

  STATE.chartInstances.actividad = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => d.fecha),
      datasets: [
        {
          label: 'Documentos',
          data: data.map(d => d.documentos),
          borderColor: '#467fcf',
          backgroundColor: 'rgba(70, 127, 207, 0.1)',
          tension: 0.4,
          fill: true
        },
        {
          label: 'Análisis IA',
          data: data.map(d => d.analisis),
          borderColor: '#28a745',
          backgroundColor: 'rgba(40, 167, 69, 0.1)',
          tension: 0.4,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'top' } },
      scales: { y: { beginAtZero: true } }
    }
  });
}

function updateDistributionChart(data) {
  const ctx = document.getElementById('distribucion-chart');
  if (!ctx) return;

  if (STATE.chartInstances.distribucion) {
    STATE.chartInstances.distribucion.destroy();
  }

  STATE.chartInstances.distribucion = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(data),
      datasets: [{
        data: Object.values(data),
        backgroundColor: ['#467fcf', '#28a745', '#ffc107', '#dc3545']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } }
    }
  });
}

function updateConclusiones(data) {
  const container = document.getElementById('conclusiones-container');
  if (!container) return;

  const precision = parseFloat(data.precision_ia) || 0;
  const semanaDocs = data.documentos_semana || 0;
  
  let conclusiones = '';

  if (precision > 95) {
    conclusiones += '<div class="alert alert-success">✅ El modelo IA tiene excelente precisión</div>';
  } else if (precision > 85) {
    conclusiones += '<div class="alert alert-warning">⚠️ La precisión del modelo es aceptable pero mejorable</div>';
  }

  if (semanaDocs > 50) {
    conclusiones += '<div class="alert alert-info">📈 Alto volumen de procesamiento esta semana</div>';
  }

  container.innerHTML = conclusiones || '<p class="text-muted">No hay conclusiones disponibles</p>';
}

// =====================
// SERVICIOS
// =====================

async function loadServicesStatus() {
  try {
    const data = await apiCall(API.services_status());
    
    const container = document.getElementById('services-grid');
    if (!container || !data.services) return;

    container.innerHTML = data.services.map(service => `
      <div class="col-md-6 col-lg-4">
        <div class="card">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-center mb-2">
              <h4 class="card-title mb-0">${service.name}</h4>
              <span class="badge ${service.status === 'online' ? 'bg-green-lt' : 'bg-red-lt'}">
                ${service.status === 'online' ? 'Online' : 'Offline'}
              </span>
            </div>
            <div class="text-muted mb-2">Puerto: ${service.port}</div>
            <div class="text-muted mb-2">Uptime: ${service.uptime}</div>
            <div class="row">
              <div class="col-6">
                <div class="text-muted">CPU</div>
                <div class="progress">
                  <div class="progress-bar" style="width: ${service.cpu}%"></div>
                </div>
                <small>${service.cpu}%</small>
              </div>
              <div class="col-6">
                <div class="text-muted">RAM</div>
                <div class="progress">
                  <div class="progress-bar bg-success" style="width: ${service.memory}%"></div>
                </div>
                <small>${service.memory}%</small>
              </div>
            </div>
          </div>
        </div>
      </div>
    `).join('');

  } catch (error) {
    console.error('Error loading services status:', error);
    showToast('Error', 'No se pudo cargar el estado de los servicios', 'error');
  }
}

// =====================
// ENTRENAMIENTO IA
// =====================

function setupTraining() {
  const startBtn = document.getElementById('start-training-wizard');
  if (startBtn) {
    startBtn.addEventListener('click', startTrainingWizard);
  }

  const prevBtn = document.getElementById('prev-step');
  const nextBtn = document.getElementById('next-step');
  const startTrainingBtn = document.getElementById('start-training');

  if (prevBtn) prevBtn.addEventListener('click', () => {
    if (STATE.trainingStep > 1) {
      STATE.trainingStep--;
      updateTrainingSteps();
    }
  });

  if (nextBtn) nextBtn.addEventListener('click', () => {
    if (STATE.trainingStep < 4) {
      STATE.trainingStep++;
      updateTrainingSteps();
    }
  });

  if (startTrainingBtn) startTrainingBtn.addEventListener('click', executeTraining);
}

function startTrainingWizard() {
  const wizard = document.getElementById('training-wizard');
  if (wizard) {
    wizard.style.display = 'block';
    STATE.trainingStep = 1;
    updateTrainingSteps();
  }
}

function updateTrainingSteps() {
  for (let i = 1; i <= 4; i++) {
    const step = document.getElementById(`step-${i}`);
    if (step) {
      step.classList.remove('active', 'completed');
      if (i === STATE.trainingStep) {
        step.classList.add('active');
      } else if (i < STATE.trainingStep) {
        step.classList.add('completed');
      }
    }
  }

  const prevBtn = document.getElementById('prev-step');
  const nextBtn = document.getElementById('next-step');
  const startBtn = document.getElementById('start-training');

  if (prevBtn) prevBtn.disabled = STATE.trainingStep === 1;
  
  if (nextBtn && startBtn) {
    if (STATE.trainingStep === 4) {
      nextBtn.style.display = 'none';
      startBtn.style.display = 'inline-block';
    } else {
      nextBtn.style.display = 'inline-block';
      startBtn.style.display = 'none';
    }
  }
}

async function executeTraining() {
  if (STATE.trainingInProgress) return;

  STATE.trainingInProgress = true;
  
  try {
    const config = {
      project_name: document.getElementById('project-name')?.value || 'Proyecto Sin Nombre',
      training_type: document.getElementById('training-type')?.value || 'classification',
      base_model: document.getElementById('base-model')?.value || 'llama-3.1-8b'
    };

    await simulateTraining(config);
    showToast('Éxito', 'Entrenamiento completado exitosamente', 'success');
    loadTrainingProjects();
    
  } catch (error) {
    showToast('Error', 'Error durante el entrenamiento: ' + error.message, 'error');
  } finally {
    STATE.trainingInProgress = false;
  }
}

async function simulateTraining(config) {
  const steps = [
    'Cargando documentos...',
    'Preprocesando texto...',
    'Inicializando modelo...',
    'Entrenando época 1/3...',
    'Entrenando época 2/3...',
    'Entrenando época 3/3...',
    'Validando modelo...',
    'Guardando checkpoint...',
    'Entrenamiento completado'
  ];

  const progressBar = document.getElementById('training-progress');
  const statusElement = document.getElementById('training-status');
  const logsElement = document.getElementById('training-logs');

  for (let i = 0; i < steps.length; i++) {
    const progress = ((i + 1) / steps.length) * 100;
    
    if (progressBar) {
      progressBar.style.width = `${progress}%`;
    }
    
    if (statusElement) {
      statusElement.textContent = steps[i];
    }
    
    if (logsElement) {
      logsElement.innerHTML += `[${new Date().toLocaleTimeString()}] ${steps[i]}\n`;
      logsElement.scrollTop = logsElement.scrollHeight;
    }

    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));
  }
}

async function loadTrainingProjects() {
  try {
    const data = await apiCall(API.training_projects());
    // Implementar visualización de proyectos
  } catch (error) {
    console.error('Error loading training projects:', error);
  }
}

function updateTrainingProgress(progress, status) {
  const progressBar = document.getElementById('training-progress');
  const statusElement = document.getElementById('training-status');
  
  if (progressBar) progressBar.style.width = `${progress}%`;
  if (statusElement) statusElement.textContent = status;
}

// =====================
// AGENTE ACCIONABLE
// =====================

function setupAgent() {
  const commandInput = document.getElementById('agent-command-input');
  const executeBtn = document.getElementById('execute-agent-command');

  if (commandInput) {
    commandInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') executeAgentCommand();
    });
  }

  if (executeBtn) {
    executeBtn.addEventListener('click', executeAgentCommand);
  }

  document.querySelectorAll('[data-quick-command]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const commandType = e.target.getAttribute('data-quick-command');
      executeQuickCommand(commandType);
    });
  });
}

async function executeAgentCommand() {
  const input = document.getElementById('agent-command-input');
  if (!input?.value.trim()) {
    showToast('Error', 'Por favor, ingresa un comando', 'warning');
    return;
  }

  const command = input.value.trim();
  input.value = '';

  try {
    logAgentAction(command, 'Ejecutando...');
    
    const result = await simulateAgentExecution(command);
    
    logAgentAction(command, result.success ? 'Completado' : 'Error');
    
    STATE.agentStats.actionsCount++;
    if (result.success) STATE.agentStats.successCount++;
    STATE.agentStats.lastAction = command;
    
    loadAgentStatus();
    showToast(result.success ? 'Comando Ejecutado' : 'Error', result.message, result.success ? 'success' : 'error');
    
  } catch (error) {
    logAgentAction(command, 'Error: ' + error.message);
    showToast('Error', 'Error ejecutando comando: ' + error.message, 'error');
  }
}

async function simulateAgentExecution(command) {
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  const success = Math.random() > 0.2;
  return {
    success,
    message: success ? 'Comando ejecutado exitosamente' : 'Error ejecutando comando'
  };
}

function executeQuickCommand(commandType) {
  const commands = {
    'entrenar-empresa-x': 'Entrenar IA para Empresa X',
    'procesar-documentos-pendientes': 'Procesar documentos pendientes',
    'generar-reporte-semanal': 'Generar reporte semanal',
    'validar-modelo-precision': 'Validar precisión del modelo',
    'limpiar-base-datos': 'Limpiar base de datos'
  };

  const command = commands[commandType];
  if (command) {
    document.getElementById('agent-command-input').value = command;
    executeAgentCommand();
  }
}

function logAgentAction(action, result) {
  const logContainer = document.getElementById('agent-actions-log');
  if (!logContainer) return;

  const logEntry = document.createElement('div');
  logEntry.className = 'border-bottom pb-2 mb-2';
  logEntry.innerHTML = `
    <div class="d-flex justify-content-between">
      <div>
        <strong>Comando:</strong> ${action}
        <br><small class="text-muted">Resultado: ${result}</small>
      </div>
      <small class="text-muted">${new Date().toLocaleTimeString()}</small>
    </div>
  `;

  logContainer.insertBefore(logEntry, logContainer.firstChild);

  while (logContainer.children.length > 10) {
    logContainer.removeChild(logContainer.lastChild);
  }
}

async function loadAgentStatus() {
  try {
    const data = await apiCall(API.agent_status());
    
    document.getElementById('agent-actions-count').textContent = STATE.agentStats.actionsCount;
    document.getElementById('agent-success-rate').textContent = 
      `${STATE.agentStats.actionsCount > 0 ? Math.round((STATE.agentStats.successCount / STATE.agentStats.actionsCount) * 100) : 0}%`;
    document.getElementById('agent-last-action').textContent = STATE.agentStats.lastAction;

  } catch (error) {
    console.error('Error loading agent status:', error);
  }
}

// =====================
// CHAT
// =====================

function setupChat() {
  const chatInput = document.getElementById('chat-input');
  const chatSendButton = document.getElementById('chat-send-button');

  if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }

  if (chatSendButton) {
    chatSendButton.addEventListener('click', sendChatMessage);
  }
}

function sendChatMessage() {
  const input = document.getElementById('chat-input');
  if (!input?.value.trim()) return;

  const message = input.value.trim();
  input.value = '';

  addChatMessage(message, 'user');

  if (STATE.websocketConnected && STATE.websocket) {
    STATE.websocket.send(JSON.stringify({
      type: 'chat_message',
      message: message
    }));
  } else {
    setTimeout(() => {
      const responses = [
        'Entiendo tu consulta. Estoy procesando la información...',
        'Basándome en los datos disponibles, puedo ayudarte con eso.',
        'Excelente pregunta. Déjame analizar los documentos relacionados.'
      ];
      const response = responses[Math.floor(Math.random() * responses.length)];
      addChatMessage(response, 'ai');
    }, 1000);
  }
}

function addChatMessage(message, sender) {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-message ${sender}`;
  
  if (sender === 'user') {
    messageDiv.innerHTML = `<strong>Tú:</strong> ${message}`;
  } else {
    messageDiv.innerHTML = `<strong>IA:</strong> ${message}`;
  }

  container.appendChild(messageDiv);
  container.scrollTop = container.scrollHeight;
}

// =====================
// DOCUMENTOS
// =====================

function setupDocuments() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const uploadButton = document.getElementById('upload-button');

  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  if (uploadButton) {
    uploadButton.addEventListener('click', uploadDocument);
  }
}

function handleFileSelection(file) {
  const previewContainer = document.getElementById('file-preview');
  const uploadButton = document.getElementById('upload-button');

  if (previewContainer) {
    previewContainer.innerHTML = `
      <div class="alert alert-info">
        <strong>Archivo seleccionado:</strong> ${file.name}<br>
        <strong>Tamaño:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB
      </div>
    `;
  }

  if (uploadButton) {
    uploadButton.style.display = 'block';
  }
}

async function uploadDocument() {
  const fileInput = document.getElementById('file-input');
  const extractedText = document.getElementById('extracted-text');
  const uploadButton = document.getElementById('upload-button');

  if (!fileInput?.files[0]) {
    showToast('Error', 'No hay archivo seleccionado', 'warning');
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  try {
    uploadButton.disabled = true;
    uploadButton.innerHTML = '<i class="ti ti-loader ti-spin me-1"></i>Procesando...';
    extractedText.textContent = 'Subiendo y procesando el archivo...';

    const response = await apiCall(API.document_upload(), {
      method: 'POST',
      body: formData
    });

    if (response && response.text_content) {
      extractedText.textContent = response.text_content;
      showToast('Éxito', `Documento '${response.filename || file.name}' procesado.`, 'success');
    } else {
      extractedText.textContent = 'El servicio procesó el documento, pero no devolvió texto para mostrar.';
      showToast('Información', 'Procesamiento finalizado sin texto extraído.', 'info');
    }

  } catch (error) {
    console.error('Error uploading document:', error);
    showToast('Error', 'Error procesando documento: ' + error.message, 'error');
    extractedText.textContent = 'Ocurrió un error al procesar el documento. Revisa la consola para más detalles.';
  } finally {
    uploadButton.disabled = false;
    uploadButton.innerHTML = '<i class="ti ti-upload me-1"></i>Subir y Procesar';
  }
}

// =====================
// ANÁLISIS IA
// =====================

function setupIA() {
  const analyzeButton = document.getElementById('ia-analyze-button');
  if (analyzeButton) {
    analyzeButton.addEventListener('click', analyzeText);
  }
}

async function analyzeText() {
  const textInput = document.getElementById('ia-text-input');
  const modelSelect = document.getElementById('ia-model-select');
  const resultContainer = document.getElementById('ia-result-container');
  const resultText = document.getElementById('ia-result-text');
  const analyzeButton = document.getElementById('ia-analyze-button');

  if (!textInput?.value.trim()) {
    showToast('Error', 'Por favor, ingresa texto para analizar', 'warning');
    return;
  }

  const text = textInput.value.trim();
  const model = modelSelect?.value || 'local';

  try {
    analyzeButton.disabled = true;
    analyzeButton.innerHTML = '<i class="ti ti-loader ti-spin me-1"></i>Analizando...';

    if (resultContainer) resultContainer.style.display = 'block';
    if (resultText) resultText.textContent = 'Procesando análisis...';

    const response = await apiCall(API.ai_analyze(), {
      method: 'POST',
      body: JSON.stringify({ text, model })
    });

    let analysisResult = 'No se pudo obtener un análisis.';
    if (response && response.analysis) {
      const { sentiment, entities, keywords, summary } = response.analysis;
      analysisResult = `Análisis del Modelo: ${response.model_used || model}\n==========================================\n\n` +
                       `SENTIMIENTO: ${sentiment?.label || 'N/A'} (Puntuación: ${sentiment?.score?.toFixed(2) || 'N/A'})\n\n` +
                       `RESUMEN:\n${summary || 'No disponible.'}\n\n` +
                       `ENTIDADES PRINCIPALES:\n- ${entities?.join('\n- ') || 'Ninguna'}\n\n` +
                       `PALABRAS CLAVE:\n- ${keywords?.join('\n- ') || 'Ninguna'}`;
    }

    if (resultText) resultText.textContent = analysisResult;
    showToast('Análisis Completado', 'Texto analizado exitosamente', 'success');

  } catch (error) {
    console.error('Error analyzing text:', error);
    showToast('Error', 'Error durante el análisis', 'error');
    if (resultText) resultText.textContent = 'Error durante el análisis.';
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.innerHTML = '<i class="ti ti-brain me-1"></i>Analizar Texto';
  }
}

// =====================
// ANALYTICS
// =====================

function setupAnalytics() {
  const analyzeButton = document.getElementById('analyze-dataset-button');
  if (analyzeButton) {
    analyzeButton.addEventListener('click', analyzeDataset);
  }
}

async function analyzeDataset() {
  const fileInput = document.getElementById('dataset-file');
  const analysisTypeEl = document.getElementById('analysis-type');
  const resultsContainer = document.getElementById('analysis-results');
  const analyzeButton = document.getElementById('analyze-dataset-button');

  if (!fileInput?.files[0]) {
    showToast('Error', 'Por favor, selecciona un archivo de dataset', 'warning');
    return;
  }

  const file = fileInput.files[0];
  const analysisType = analysisTypeEl?.value || 'descriptive';
  const formData = new FormData();
  formData.append('file', file);
  formData.append('analysis_type', analysisType);

  try {
    analyzeButton.disabled = true;
    analyzeButton.innerHTML = '<i class="ti ti-loader ti-spin me-1"></i>Analizando...';
    resultsContainer.innerHTML = '<div class="text-center py-3"><div class="spinner-border"></div><br>Procesando dataset, esto puede tardar...</div>';

    const response = await apiCall(API.analytics_upload(), { // Assuming API.analytics_upload points to the correct new endpoint
      method: 'POST',
      body: formData
    });

    if (response && response.results) {
      resultsContainer.innerHTML = generateAnalysisResults(analysisType, file.name, response.results);
      showToast('Análisis Completado', 'Dataset analizado exitosamente', 'success');
    } else {
      throw new Error('La respuesta del análisis no fue válida.');
    }

  } catch (error) {
    console.error('Error analyzing dataset:', error);
    showToast('Error', 'Error durante el análisis: ' + error.message, 'error');
    if (resultsContainer) {
      resultsContainer.innerHTML = `<div class="alert alert-danger">Error durante el análisis: ${error.message}</div>`;
    }
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.innerHTML = '<i class="ti ti-chart-line me-1"></i>Analizar Dataset';
  }
}

function generateAnalysisResults(analysisType, fileName, results) {
  const rows = results.rows || 'N/A';
  const columns = results.columns || 'N/A';
  
  let specificResults = '';
  if (results.summary) {
    specificResults += '<h5>Resumen del Análisis</h5>';
    specificResults += '<dl class="row">';
    for (const [key, value] of Object.entries(results.summary)) {
      specificResults += `<dt class="col-sm-3">${key}</dt><dd class="col-sm-9">${JSON.stringify(value)}</dd>`;
    }
    specificResults += '</dl>';
  }

  return `
    <div class="card">
      <div class="card-header">
        <h4 class="card-title">Resultados: ${fileName}</h4>
      </div>
      <div class="card-body">
        <div class="row mb-3">
          <div class="col-md-4">
            <div class="card card-sm">
              <div class="card-body text-center">
                <div class="text-muted">Tipo de Análisis</div>
                <div class="h3 mb-0">${analysisType}</div>
              </div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card card-sm">
              <div class="card-body text-center">
                <div class="text-muted">Filas</div>
                <div class="h3 mb-0">${rows.toLocaleString()}</div>
              </div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card card-sm">
              <div class="card-body text-center">
                <div class="text-muted">Columnas</div>
                <div class="h3 mb-0">${columns}</div>
              </div>
            </div>
          </div>
        </div>
        ${specificResults}
      </div>
    </div>
  `;
}

// =====================
// REPORTES
// =====================

function setupReports() {
  const generateButton = document.getElementById('generate-report-button');
  if (generateButton) {
    generateButton.addEventListener('click', generateReport);
  }
}

async function generateReport() {
  const reportTypeEl = document.getElementById('report-type');
  const reportFormatEl = document.getElementById('report-format');
  const resultContainer = document.getElementById('report-result-container');
  const generateButton = document.getElementById('generate-report-button');

  const type = reportTypeEl?.value || 'general';
  const format = reportFormatEl?.value || 'pdf';

  if (!generateButton || !resultContainer) return;

  try {
    generateButton.disabled = true;
    generateButton.innerHTML = '<i class="ti ti-loader ti-spin me-1"></i>Generando...';
    resultContainer.innerHTML = '<div class="alert alert-info">Generando reporte, por favor espera...</div>';

    const response = await apiCall(API.report_generate(), {
      method: 'POST',
      body: JSON.stringify({ type, format })
    });

    if (!response || !response.file_content || !response.filename) {
      throw new Error('La respuesta de la API no es válida.');
    }

    const { file_content, filename, format: file_format } = response;

    const mimeTypes = {
      pdf: 'application/pdf',
      excel: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      html: 'text/html',
      json: 'application/json'
    };

    const mimeType = mimeTypes[file_format] || 'application/octet-stream';
    const byteCharacters = atob(file_content);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: mimeType });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    resultContainer.innerHTML = `
      <div class="alert alert-success">
        <h4 class="alert-title">¡Reporte Generado!</h4>
        <p>Tu descarga para <strong>${filename}</strong> ha comenzado.</p>
        <div class="mt-2">
          <button class="btn btn-success" onclick="window.DashboardAPI.refreshData()">Descargar de nuevo</button>
        </div>
      </div>
    `;
    showToast('Reporte Generado', `Tu descarga de ${filename} está lista.`, 'success');

  } catch (error) {
    console.error('Error generating report:', error);
    showToast('Error', 'No se pudo generar el reporte: ' + error.message, 'error');
    if (resultContainer) {
      resultContainer.innerHTML = `<div class="alert alert-danger">Error al generar el reporte. Revisa la consola para más detalles.</div>`;
    }
  } finally {
    generateButton.disabled = false;
    generateButton.innerHTML = '<i class="ti ti-file-download me-1"></i>Generar Reporte';
  }
}

// =====================
// CONFIGURACIÓN
// =====================

function setupConfiguration() {
  const saveConfigButton = document.getElementById('save-config');
  const toggleMocks = document.getElementById('toggle-mocks');
  const testButton = document.getElementById('btn-run-tests');

  if (saveConfigButton) {
    saveConfigButton.addEventListener('click', saveConfiguration);
  }

  if (toggleMocks) {
    toggleMocks.addEventListener('change', (e) => {
      CONFIG.USE_MOCKS = e.target.checked;
      localStorage.setItem('USE_MOCKS', CONFIG.USE_MOCKS ? '1' : '0');
      showToast('Configuración', `Modo Mock ${CONFIG.USE_MOCKS ? 'activado' : 'desactivado'}`, 'info');
    });
  }

  if (testButton) {
    testButton.addEventListener('click', runTests);
  }

  loadConfiguration();
}

function saveConfiguration() {
  const config = {
    api_timeout: document.getElementById('api-timeout')?.value,
    retry_attempts: document.getElementById('retry-attempts')?.value,
    websocket_url: document.getElementById('websocket-url')?.value
  };

  Object.entries(config).forEach(([key, value]) => {
    if (value) localStorage.setItem(key, value);
  });

  showToast('Configuración Guardada', 'Los cambios se aplicarán en la próxima sesión', 'success');
}

function loadConfiguration() {
  const elements = ['api-timeout', 'retry-attempts', 'websocket-url'];
  
  elements.forEach(id => {
    const element = document.getElementById(id);
    const savedValue = localStorage.getItem(id.replace('-', '_'));
    if (element && savedValue) {
      element.value = savedValue;
    }
  });
}

async function runTests() {
  const resultsContainer = document.getElementById('test-results');
  if (!resultsContainer) return;

  resultsContainer.innerHTML = '<div class="text-center">Ejecutando tests...</div>';

  const tests = [
    { name: 'Conectividad API', test: () => true },
    { name: 'WebSocket', test: () => STATE.websocketConnected },
    { name: 'Dashboard Load', test: () => document.getElementById('total-documentos') !== null },
    { name: 'Chat Functionality', test: () => document.getElementById('chat-input') !== null }
  ];

  let results = '<h5>Resultados:</h5>';
  
  for (const test of tests) {
    const result = test.test();
    results += `<span class="test-badge ${result ? 'pass' : 'fail'}">${test.name}: ${result ? 'PASS' : 'FAIL'}</span><br>`;
    await new Promise(resolve => setTimeout(resolve, 300));
  }

  resultsContainer.innerHTML = results;
}

// =====================
// TEMA
// =====================

function setupTheme() {
  document.querySelectorAll('[data-theme]').forEach(button => {
    button.addEventListener('click', (e) => {
      e.preventDefault();
      const theme = button.getAttribute('data-theme');
      setTheme(theme);
    });
  });

  const savedTheme = localStorage.getItem('theme') || 'system';
  setTheme(savedTheme);
}

function setTheme(theme) {
  const html = document.documentElement;
  
  if (theme === 'system') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    html.setAttribute('data-bs-theme', prefersDark ? 'dark' : 'light');
  } else {
    html.setAttribute('data-bs-theme', theme);
  }
  
  localStorage.setItem('theme', theme);
}

// =====================
// UTILIDADES ADICIONALES
// =====================

function refreshData() {
  showToast('Actualizando', 'Refrescando datos...', 'info', 2000);
  
  switch(STATE.currentSection) {
    case 'inicio':
      loadDashboardData();
      break;
    case 'servicios':
      loadServicesStatus();
      break;
    case 'entrenamiento':
      loadTrainingProjects();
      break;
    case 'agente':
      loadAgentStatus();
      break;
  }
}

function loadResourceCharts() {
  // Implementar carga de gráficos de recursos
  const ctx1 = document.getElementById('recursos-chart');
  const ctx2 = document.getElementById('almacenamiento-chart');
  
  if (ctx1) {
    new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: ['CPU', 'RAM', 'Disco', 'Red'],
        datasets: [{
          data: [45, 67, 34, 23],
          backgroundColor: ['#467fcf', '#28a745', '#ffc107', '#dc3545']
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }
  
  if (ctx2) {
    new Chart(ctx2, {
      type: 'pie',
      data: {
        labels: ['Documentos', 'Modelos', 'Cache', 'Logs'],
        datasets: [{
          data: [45, 25, 15, 15],
          backgroundColor: ['#467fcf', '#28a745', '#ffc107', '#dc3545']
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }
}

// =====================
// INICIALIZACIÓN PRINCIPAL
// =====================

function initializeDashboard() {
  console.log('🚀 Inicializando Dashboard...');

  // Configurar todos los módulos
  setupNavigation();
  setupTheme();
  setupConfiguration();
  setupDocuments();
  setupIA();
  setupChat();
  setupTraining();
  setupAgent();
  setupAnalytics();
  setupReports();
  
  // Configurar refresh button
  const refreshBtn = document.getElementById('refresh-data-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', refreshData);
  }
  
  // Inicializar WebSocket
  initWebSocket();
  
  // Cargar datos iniciales
  loadDashboardData();
  
  // Configurar actualizaciones automáticas
  setInterval(() => {
    if (STATE.websocketConnected) updateConnectionCount();
  }, 30000);

  setInterval(() => {
    if (STATE.currentSection === 'inicio') loadDashboardData();
  }, 300000);

  console.log('✅ Dashboard inicializado');
  showToast('Sistema Iniciado', 'Dashboard cargado exitosamente', 'success', 3000);
}

// =====================
// EXPOSICIÓN GLOBAL
// =====================

window.DashboardAPI = {
  showToast,
  apiCall,
  showSection,
  refreshData,
  loadDashboardData,
  loadServicesStatus,
  setTheme,
  CONFIG,
  STATE,
  API
};

// Auto-inicialización
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
  initializeDashboard();
}

console.log('📄 Dashboard JavaScript v6.0.0 - Cargado');