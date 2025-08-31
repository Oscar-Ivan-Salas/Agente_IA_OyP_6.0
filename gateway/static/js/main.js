'use strict';

// ====================================
// AGENTE IA OYP 6.0 - MAIN JAVASCRIPT
// ====================================

document.addEventListener('DOMContentLoaded', function() {
  console.log("🚀 Inicializando Agente IA OyP 6.0...");

  // --- STATE MANAGEMENT ---
  const state = {
    activeProject: localStorage.getItem('activeProject') || null,
    activeJobId: null,
    jobPollingInterval: null,
    ws: null,
    wsConnected: false,
  };

  // --- DOM ELEMENTS CACHE ---
  const ui = {
    pageTitle: document.getElementById('page-title'),
    alertsContainer: document.getElementById('alerts-container'),
    wsIndicator: document.getElementById('ws-indicator'),
    // Document Section Elements
    docProjectSelector: document.getElementById('doc-project-selector'), // Assuming a <select> element
    docNewProjectBtn: document.getElementById('doc-new-project-btn'), // Assuming a button
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    uploadButton: document.getElementById('upload-button'),
    filePreview: document.getElementById('file-preview'),
    rawFilesList: document.getElementById('doc-raw-files-list'), // Assuming a <ul> or <div>
    cleanFilesList: document.getElementById('doc-clean-files-list'),
    jsonFilesList: document.getElementById('doc-json-files-list'),
    docPlanGoals: document.getElementById('doc-plan-goals'), // Assuming a <textarea>
    docPlanBtn: document.getElementById('doc-plan-btn'), // Assuming a button
    docPlanResult: document.getElementById('doc-plan-result'), // Assuming a <div>
    docRunBtn: document.getElementById('doc-run-btn'), // Assuming a button
    docJobStatus: document.getElementById('doc-job-status'), // Assuming a <div>
    docExportBtn: document.getElementById('doc-export-btn'), // Assuming a button
    // IA Section Elements
    iaAnalyzeButton: document.getElementById('ia-analyze-button'),
    iaTextInput: document.getElementById('ia-text-input'),
    iaResultContainer: document.getElementById('ia-result-container'),
    iaResultText: document.getElementById('ia-result-text'),
  };

  // --- UTILS ---
  const showToast = (message, type = 'info') => {
    // Simple alert for now, can be replaced with a proper toast library
    alert(`${type.toUpperCase()}: ${message}`);
  };

  const api = async (service, endpoint, options = {}) => {
    const url = `/api/services/${service}/${endpoint}`;
    const headers = { 'Content-Type': 'application/json', ...options.headers };
    if (options.body && !(options.body instanceof FormData)) {
      options.body = JSON.stringify(options.body);
    }

    try {
      const response = await fetch(url, { ...options, headers });
      const responseData = await response.json();
      if (!responseData.ok) {
        throw new Error(responseData.error?.message || 'Error en la respuesta del API');
      }
      return responseData.data;
    } catch (error) {
      showToast(`Error en API: ${error.message}`, 'danger');
      console.error(`API Error calling ${url}:`, error);
      throw error;
    }
  };

  // --- NAVIGATION ---
  function setupNavigation() {
    document.querySelectorAll('[data-target]').forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        showSection(link.dataset.target);
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
      });
    });
  }

  function showSection(sectionId) {
    document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
    const section = document.getElementById(sectionId);
    if (section) {
      section.classList.add('active');
      ui.pageTitle.textContent = document.querySelector(`.nav-link[data-target="${sectionId}"] .nav-link-title`).textContent;
    }
  }

  // --- DOCUMENT PROCESSOR SECTION LOGIC ---
  async function docEnsureProject() {
    if (state.activeProject) return state.activeProject;
    try {
      const newProject = await api('document_processor', 'projects', { method: 'POST', body: { name: `proj_${Date.now()}` } });
      state.activeProject = newProject.project_id;
      localStorage.setItem('activeProject', state.activeProject);
      showToast(`Nuevo proyecto creado: ${state.activeProject}`, 'success');
      await docList('raw');
      return state.activeProject;
    } catch (error) {
      showToast('No se pudo crear un nuevo proyecto.', 'danger');
    }
  }

  async function docUpload(files) {
    const projectId = await docEnsureProject();
    if (!projectId) return;

    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }

    showToast(`Subiendo ${files.length} archivo(s)...`, 'info');
    try {
      const result = await api('document_processor', `projects/${projectId}/upload`, { method: 'POST', body: formData });
      showToast(`${result.uploaded_files.length} archivo(s) subidos con éxito.`, 'success');
      await docList('raw');
    } catch (error) {
      showToast('Error durante la subida de archivos.', 'danger');
    }
  }

  async function docList(stage) {
    if (!state.activeProject) return;
    const lists = { raw: ui.rawFilesList, clean: ui.cleanFilesList, json: ui.jsonFilesList };
    const listEl = lists[stage];
    if (!listEl) return;

    try {
      const result = await api('document_processor', `projects/${state.activeProject}/files?stage=${stage}`);
      listEl.innerHTML = result.files.map(f => `<li>${f}</li>`).join('') || '<li class="text-muted">No hay archivos.</li>';
    } catch (error) {
      listEl.innerHTML = '<li class="text-danger">Error al cargar archivos.</li>';
    }
  }

  async function docPlanWithAI() {
    const projectId = await docEnsureProject();
    if (!projectId || !ui.docPlanGoals.value) {
      showToast('Por favor, define los objetivos del pipeline.', 'warning');
      return;
    }
    try {
      const plan = await api('document_processor', `projects/${projectId}/pipeline/plan`, {
        method: 'POST',
        body: { sample_files: [], goals: ui.docPlanGoals.value.split(',').map(s => s.trim()) }
      });
      ui.docPlanResult.innerHTML = plan.suggested_steps.map(step => `<span class="badge bg-primary-lt m-1">${step.step}</span>`).join('');
      if (ui.docRunBtn) ui.docRunBtn.dataset.plan = JSON.stringify(plan.suggested_steps);
    } catch (error) {
      showToast('Error al planificar con IA.', 'danger');
    }
  }

  async function docRunPipeline() {
    const projectId = await docEnsureProject();
    const plan = ui.docRunBtn?.dataset.plan;
    if (!projectId || !plan) {
      showToast('No hay un plan para ejecutar.', 'warning');
      return;
    }
    try {
      const result = await api('document_processor', `projects/${projectId}/pipeline/run`, { method: 'POST', body: { steps: JSON.parse(plan) } });
      state.activeJobId = result.job_id;
      showToast(`Pipeline iniciado con Job ID: ${state.activeJobId}`, 'info');
      if (state.jobPollingInterval) clearInterval(state.jobPollingInterval);
      state.jobPollingInterval = setInterval(docCheckJobStatus, 2000);
    } catch (error) {
      showToast('Error al iniciar el pipeline.', 'danger');
    }
  }

  async function docCheckJobStatus() {
    if (!state.activeJobId) return;
    try {
      const job = await api('document_processor', `jobs/${state.activeJobId}`);
      if (ui.docJobStatus) ui.docJobStatus.textContent = `Estado del Job: ${job.status}, Progreso: ${(job.progress * 100).toFixed(0)}%`;
      if (job.status === 'completed') {
        clearInterval(state.jobPollingInterval);
        state.jobPollingInterval = null;
        showToast(`Job ${state.activeJobId} completado.`, 'success');
        await docList('clean');
        await docList('json');
      }
    } catch (error) {
      clearInterval(state.jobPollingInterval);
      showToast('Error al verificar el estado del job.', 'danger');
    }
  }

  function docExport() {
    if (!state.activeProject) {
      showToast('No hay un proyecto activo para exportar.', 'warning');
      return;
    }
    const url = `/api/services/document_processor/projects/${state.activeProject}/export?format=zip-json`;
    window.open(url, '_blank');
  }

  function setupDocumentsSection() {
    if (ui.dropZone) {
      ui.dropZone.addEventListener('click', () => ui.fileInput.click());
      ui.fileInput.addEventListener('change', () => docUpload(ui.fileInput.files));
      ui.dropZone.addEventListener('dragover', e => { e.preventDefault(); ui.dropZone.classList.add('dragover'); });
      ui.dropZone.addEventListener('dragleave', () => ui.dropZone.classList.remove('dragover'));
      ui.dropZone.addEventListener('drop', e => {
        e.preventDefault();
        ui.dropZone.classList.remove('dragover');
        docUpload(e.dataTransfer.files);
      });
    }
    if (ui.docPlanBtn) ui.docPlanBtn.addEventListener('click', docPlanWithAI);
    if (ui.docRunBtn) ui.docRunBtn.addEventListener('click', docRunPipeline);
    if (ui.docExportBtn) ui.docExportBtn.addEventListener('click', docExport);
    // Initial load
    if(state.activeProject) docList('raw');
  }

  // --- IA ANALYSIS SECTION ---
  async function handleTextAnalysis() {
    const text = ui.iaTextInput.value.trim();
    if (!text) return showToast('Por favor, introduce texto para analizar.', 'warning');
    try {
      const result = await api('ai_engine', 'analyze_text', { method: 'POST', body: { text } });
      ui.iaResultText.textContent = JSON.stringify(result, null, 2);
      ui.iaResultContainer.style.display = 'block';
    } catch (error) {
      showToast('Error en el análisis de IA.', 'danger');
    }
  }

  function setupIASection() {
    if (ui.iaAnalyzeButton) ui.iaAnalyzeButton.addEventListener('click', handleTextAnalysis);
  }

  // --- INITIALIZATION ---
  setupNavigation();
  setupDocumentsSection();
  setupIASection();
  showSection('inicio'); // Show initial section
  console.log("✅ Agente IA OyP 6.0 inicializado.");
});