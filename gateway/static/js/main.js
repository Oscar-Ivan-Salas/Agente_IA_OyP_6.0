'use strict';

// Configuración Global
window.AgenteIA = {
  config: {
    mock: true,
    apiTimeout: 15000,
    wsUrl: 'ws://localhost:8765',
    maxFileSize: 10 * 1024 * 1024, // 10 MB
  },
  state: {
    currentPage: 'inicio',
    ws: null,
    wsConnected: false,
    models: [],
    currentTraining: null,
  },
  utils: {
    // Función para mostrar alertas dinámicas
    showAlert(message, type = 'info', icon = 'info-circle') {
      const alertsContainer = document.getElementById('alerts-container');
      const alertId = `alert-${Date.now()}`;
      const alertHTML = `
        <div class="alert alert-${type} alert-dismissible" role="alert" id="${alertId}">
          <div class="d-flex">
            <div><i class="ti ti-${icon} me-2"></i></div>
            <div>${message}</div>
          </div>
          <a class="btn-close" data-bs-dismiss="alert" aria-label="close"></a>
        </div>`;
      alertsContainer.insertAdjacentHTML('beforeend', alertHTML);
      setTimeout(() => {
        const alertElement = document.getElementById(alertId);
        if (alertElement) alertElement.remove();
      }, 5000);
    },

    // Simulación de llamada a API
    async apiCall(endpoint, options = {}) {
      const { mock } = window.AgenteIA.config;
      const { showAlert } = window.AgenteIA.utils;

      console.log(`API Call: ${endpoint}`, { mock, ...options });

      if (mock) {
        await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));
        switch (endpoint) {
          case '/api/dashboard-stats':
            return { 
              totalDocumentos: 1578, 
              documentosSemana: 234, 
              precisionIA: '96.4%',
              actividadSemana: { labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'], data: [30, 45, 60, 50, 70, 85, 90] },
              distribucionDocumentos: { labels: ['Contratos', 'Facturas', 'Informes', 'Otros'], data: [40, 25, 20, 15] },
              conclusiones: ['Aumento del 15% en procesamiento de contratos.', 'Modelo de extracción de facturas alcanzó 98% de precisión.'],
              servicios: [{name: 'Motor IA', status: 'online'}, {name: 'Procesador Docs', status: 'online'}, {name: 'Gateway API', status: 'degraded'}]
            };
          case '/api/upload':
            return { success: true, text: `Texto extraído del documento: ${options.body.get('file').name}` };
          case '/api/analyze':
            return { success: true, result: `Análisis completo para el texto usando ${options.body.model}. Resultado: Insight clave detectado.` };
          case '/api/training/start':
            return { success: true, trainingId: `train_${Date.now()}` };
          default:
            return { success: false, message: 'Endpoint simulado no encontrado' };
        }
      }

      try {
        const response = await fetch(endpoint, {
          ...options,
          headers: { 'Content-Type': 'application/json', ...options.headers },
          body: options.body ? JSON.stringify(options.body) : null,
        });
        if (!response.ok) throw new Error(`Error en la respuesta de la API: ${response.statusText}`);
        return await response.json();
      } catch (error) {
        showAlert(`Error de conexión con la API: ${error.message}`, 'danger', 'alert-triangle');
        return { success: false, message: error.message };
      }
    }
  }
};

document.addEventListener('DOMContentLoaded', function() {
  const { config, state, utils } = window.AgenteIA;
  const { showAlert, apiCall } = utils;

  // Inicializar Navegación
  function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link[data-target]');
    const contentSections = document.querySelectorAll('.content-section');
    const pageTitle = document.getElementById('page-title');

    function switchPage(targetId) {
      contentSections.forEach(section => section.classList.remove('active'));
      navLinks.forEach(link => link.classList.remove('active'));

      const targetSection = document.getElementById(targetId);
      const targetLink = document.querySelector(`.nav-link[data-target="${targetId}"]`);

      if (targetSection) targetSection.classList.add('active');
      if (targetLink) {
        targetLink.classList.add('active');
        pageTitle.textContent = targetLink.querySelector('.nav-link-title').textContent;
      }
      state.currentPage = targetId;
      console.log(`Página cambiada a: ${targetId}`);
    }

    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        switchPage(link.dataset.target);
      });
    });
    
    document.body.addEventListener('click', e => {
      if (e.target.closest('[data-go]')) {
        e.preventDefault();
        const target = e.target.closest('[data-go]').dataset.go;
        switchPage(target);
      }
    });
  }

  // Inicializar Tema
  function setupTheme() {
    const themeToggle = document.querySelector('.dropdown[title="Cambiar tema"]');
    const currentThemeIcon = themeToggle.querySelector('i');
    const themes = {
      light: 'ti-sun-high',
      dark: 'ti-moon',
      system: 'ti-device-desktop'
    };

    function applyTheme(theme) {
      document.documentElement.setAttribute('data-bs-theme', theme);
      currentThemeIcon.className = `ti ${themes[theme]}`;
      localStorage.setItem('theme', theme);
    }

    themeToggle.querySelectorAll('.dropdown-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const selectedTheme = e.currentTarget.dataset.theme;
        if (selectedTheme === 'system') {
          localStorage.removeItem('theme');
          const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
          applyTheme(systemTheme);
        } else {
          applyTheme(selectedTheme);
        }
      });
    });

    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      applyTheme(savedTheme);
    } else {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      applyTheme(systemTheme);
    }
  }

  // Conexión WebSocket
  function setupWebSocket() {
    const wsIndicator = document.getElementById('ws-indicator');
    
    function connect() {
      state.ws = new WebSocket(config.wsUrl);
      wsIndicator.classList.remove('connected');
      wsIndicator.classList.add('connecting');

      state.ws.onopen = () => {
        state.wsConnected = true;
        wsIndicator.classList.remove('connecting');
        wsIndicator.classList.add('connected');
        wsIndicator.title = 'WebSocket Conectado';
        showAlert('Conectado al servidor en tiempo real.', 'success', 'wifi');
        document.getElementById('conexiones-ws').textContent = '1';
      };

      state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('WS Message:', data);
        // Aquí se manejarían los mensajes del servidor
      };

      state.ws.onclose = () => {
        state.wsConnected = false;
        wsIndicator.className = 'ws-status';
        wsIndicator.title = 'WebSocket Desconectado';
        showAlert('Conexión en tiempo real perdida. Intentando reconectar...', 'warning', 'wifi-off');
        document.getElementById('conexiones-ws').textContent = '0';
        setTimeout(connect, 5000);
      };

      state.ws.onerror = (error) => {
        showAlert('Error en la conexión WebSocket.', 'danger', 'alert-triangle');
        console.error('WebSocket Error:', error);
        state.ws.close();
      };
    }
    connect();
  }

  // Cargar datos del Dashboard
  async function loadDashboardData() {
    const data = await apiCall('/api/dashboard-stats');
    if (!data) return;

    document.getElementById('total-documentos').textContent = data.totalDocumentos;
    document.getElementById('documentos-semana').textContent = data.documentosSemana;
    document.getElementById('precision-ia').textContent = data.precisionIA;

    // Gráfico de Actividad
    new Chart(document.getElementById('actividad-chart'), {
      type: 'line',
      data: {
        labels: data.actividadSemana.labels,
        datasets: [{
          label: 'Documentos Procesados',
          data: data.actividadSemana.data,
          borderColor: '#467fcf',
          tension: 0.4,
          fill: true,
          backgroundColor: 'rgba(70,127,207,0.1)'
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });

    // Gráfico de Distribución
    new Chart(document.getElementById('distribucion-chart'), {
      type: 'doughnut',
      data: {
        labels: data.distribucionDocumentos.labels,
        datasets: [{
          data: data.distribucionDocumentos.data,
          backgroundColor: ['#467fcf', '#2fb344', '#f9a825', '#677788']
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });

    // Conclusiones y Servicios
    const conclusionesContainer = document.getElementById('conclusiones-container');
    conclusionesContainer.innerHTML = data.conclusiones.map(c => `<p><i class="ti ti-check text-success me-2"></i>${c}</p>`).join('');
    const serviciosContainer = document.getElementById('servicios-container');
    serviciosContainer.innerHTML = data.servicios.map(s => `<div class="d-flex justify-content-between"><span>${s.name}</span><span class="badge bg-${s.status === 'online' ? 'success' : 'warning'}-lt">${s.status}</span></div>`).join('');
  }

  // Sección de Documentos
  function setupDocumentUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const filePreview = document.getElementById('file-preview');
    const extractedText = document.getElementById('extracted-text');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
      if (!file) return;
      if (file.size > config.maxFileSize) {
        showAlert('El archivo es demasiado grande.', 'danger');
        return;
      }
      selectedFile = file;
      filePreview.innerHTML = `<p>Archivo seleccionado: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(2)} KB)</p>`;
      uploadButton.style.display = 'inline-block';
    }

    uploadButton.addEventListener('click', async () => {
      if (!selectedFile) return;
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      uploadButton.disabled = true;
      uploadButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Procesando...';

      const response = await apiCall('/api/upload', { method: 'POST', body: formData });
      
      if (response.success) {
        extractedText.textContent = response.text;
        showAlert('Documento procesado con éxito.', 'success');
      } else {
        showAlert('Error al procesar el documento.', 'danger');
      }

      uploadButton.disabled = false;
      uploadButton.innerHTML = '<i class="ti ti-upload me-1"></i>Subir y Procesar';
    });
  }

  // Sección de IA
  function setupIAAnalysis() {
    const analyzeButton = document.getElementById('ia-analyze-button');
    const textInput = document.getElementById('ia-text-input');
    const modelSelect = document.getElementById('ia-model-select');
    const resultContainer = document.getElementById('ia-result-container');
    const resultText = document.getElementById('ia-result-text');

    analyzeButton.addEventListener('click', async () => {
      const text = textInput.value.trim();
      if (!text) {
        showAlert('Por favor, introduce texto para analizar.', 'warning');
        return;
      }

      analyzeButton.disabled = true;
      analyzeButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Analizando...';

      const response = await apiCall('/api/analyze', {
        method: 'POST',
        body: { text, model: modelSelect.value }
      });

      if (response.success) {
        resultText.textContent = response.result;
        resultContainer.style.display = 'block';
      } else {
        showAlert('Error en el análisis de IA.', 'danger');
      }

      analyzeButton.disabled = false;
      analyzeButton.innerHTML = '<i class="ti ti-brain me-1"></i>Analizar Texto';
    });
  }

  // Inicialización de todas las funciones
  setupNavigation();
  setupTheme();
  setupWebSocket();
  loadDashboardData();
  setupDocumentUpload();
  setupIAAnalysis();

  console.log('Agente IA OyP 6.0 inicializado.');
});
