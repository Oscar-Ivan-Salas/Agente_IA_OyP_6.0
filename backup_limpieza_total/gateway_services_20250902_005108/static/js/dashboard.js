'use strict';

// Initialize globals
const { api, showMsg } = window.AgenteIA || {};

// Chart instances cache
const chartInstances = new Map();

/**
 * Initialize all dashboard components
 */
async function initDashboard() {
  console.log('🚀 Inicializando Dashboard...');
  
  try {
    // Initialize charts
    await initCharts();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load initial data
    await loadDashboardData();
    
    console.log('✅ Dashboard inicializado correctamente');
  } catch (error) {
    console.error('❌ Error al inicializar el dashboard:', error);
    window.AgenteIA.utils.showAlert('Error al cargar el dashboard. Por favor, recarga la página.', 'danger');
  }
}

/**
 * Initialize all charts
 */
async function initCharts() {
  // Activity Chart
  const activityCtx = document.getElementById('activity-chart');
  if (activityCtx) {
    const activityChart = new Chart(activityCtx.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Actividad',
          data: [],
          borderColor: '#206bc4',
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: false }
        },
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
    chartInstances.set('activity', activityChart);
  }
  
  // Add more chart initializations as needed
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
  // Document upload and extraction
  const fileInput = document.getElementById('document-folder') || document.getElementById('document-preview');
  const docType = document.getElementById('document-type-filter');

  // File upload handler
  if (fileInput) {
    fileInput.addEventListener('change', handleFileUpload);
  }

  // AI Analyze button
  const analyzeBtn = document.getElementById('ia-analyze-button');
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', handleAIAnalysis);
  }

  // Report generation
  const reportBtn = document.getElementById('generate-report-button');
  if (reportBtn) {
    reportBtn.addEventListener('click', generateReport);
  }

  // Refresh data
  const refreshBtn = document.getElementById('refresh-data-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', refreshDashboardData);
  }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initDashboard();
});

// File upload handler
async function handleFileUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${window.AgenteIA.config.API_BASE}/api/documents/upload`, {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    if (!response.ok) {
      throw new Error(result.error?.message || 'Error al cargar el archivo');
    }
    
    window.AgenteIA.utils.showAlert('Archivo cargado exitosamente', 'success');
    
    // Update UI with the uploaded file info
    updateFilePreview(file, result.data);
    
  } catch (error) {
    console.error('Error uploading file:', error);
    window.AgenteIA.utils.showAlert(error.message, 'danger');
  }
}

// Handle AI Analysis
async function handleAIAnalysis() {
  const inputText = document.getElementById('ia-text-input')?.value?.trim();
  const mode = document.getElementById('ia-model-select')?.value || 'search';
  
  if (!inputText) {
    window.AgenteIA.utils.showAlert('Por favor ingresa un texto para analizar', 'warning');
    return;
  }

  try {
    let result;
    
    if (mode === 'search') {
      result = await fetch(`${window.AgenteIA.config.API_BASE}/api/ai/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: inputText, k: 4 })
      });
    } else if (mode === 'ner') {
      result = await fetch(`${window.AgenteIA.config.API_BASE}/api/ai/ner`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText })
      });
    }
    
    const data = await result.json();
    
    if (!result.ok) {
      throw new Error(data.error?.message || 'Error en el análisis');
    }
    
    displayAnalysisResults(data, mode);
    
  } catch (error) {
    console.error('Error during AI analysis:', error);
    window.AgenteIA.utils.showAlert(error.message, 'danger');
  }
}

// Generate report
async function generateReport() {
  const reportType = document.getElementById('report-type')?.value || 'weekly';
  const projectName = document.getElementById('project-name')?.value || 'default';
  
  try {
    const endpoint = reportType === 'weekly' ? 'weekly_report' : 'final_report';
    const response = await fetch(
      `${window.AgenteIA.config.API_BASE}/api/projects/${encodeURIComponent(projectName)}/${endpoint}.docx`
    );
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Error al generar el reporte');
    }
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    
    // Create download link
    const a = document.createElement('a');
    a.href = url;
    a.download = `${reportType}_${projectName}_${new Date().toISOString().split('T')[0]}.docx`;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    window.URL.revokeObjectURL(url);
    a.remove();
    
    window.AgenteIA.utils.showAlert('Reporte generado exitosamente', 'success');
    
  } catch (error) {
    console.error('Error generating report:', error);
    window.AgenteIA.utils.showAlert(error.message, 'danger');
  }
}

// Refresh dashboard data
async function refreshDashboardData() {
  try {
    // Show loading state
    const refreshBtn = document.getElementById('refresh-data-btn');
    const originalText = refreshBtn?.innerHTML;
    
    if (refreshBtn) {
      refreshBtn.disabled = true;
      refreshBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Actualizando...';
    }
    
    // Refresh charts
    await updateCharts();
    
    // Update other dashboard data
    await loadDashboardData();
    
    window.AgenteIA.utils.showAlert('Datos actualizados correctamente', 'success');
    
  } catch (error) {
    console.error('Error refreshing dashboard data:', error);
    window.AgenteIA.utils.showAlert('Error al actualizar los datos', 'danger');
    
  } finally {
    // Restore button state
    const refreshBtn = document.getElementById('refresh-data-btn');
    if (refreshBtn) {
      refreshBtn.disabled = false;
      refreshBtn.innerHTML = originalText || 'Actualizar Datos';
    }
  }
}

// Update charts with new data
async function updateCharts() {
  try {
    // Example: Fetch new data for activity chart
    const response = await fetch(`${window.AgenteIA.config.API_BASE}/api/analytics/activity`);
    const data = await response.json();
    
    if (response.ok && data.data) {
      const activityChart = chartInstances.get('activity');
      if (activityChart) {
        activityChart.data.labels = data.data.labels || [];
        activityChart.data.datasets[0].data = data.data.values || [];
        activityChart.update();
      }
    }
    
    // Add more chart updates as needed
    
  } catch (error) {
    console.error('Error updating charts:', error);
    throw error;
  }
}

// Load initial dashboard data
async function loadDashboardData() {
  try {
    await updateCharts();
    
    // Add more data loading as needed
    
  } catch (error) {
    console.error('Error loading dashboard data:', error);
    throw error;
  }
}

// Update file preview in the UI
function updateFilePreview(file, fileData) {
  const previewArea = document.getElementById('document-preview') || document.getElementById('extracted-text');
  if (!previewArea) return;
  
  // Show basic file info
  previewArea.innerHTML = `
    <div class="card">
      <div class="card-body">
        <h5 class="card-title">${file.name}</h5>
        <p class="text-muted">${(file.size / 1024).toFixed(2)} KB</p>
        ${fileData?.text ? 
          `<div class="mt-3">
            <h6>Contenido extraído:</h6>
            <div class="p-2 bg-light rounded" style="max-height: 300px; overflow-y: auto;">
              ${fileData.text.slice(0, 1000)}${fileData.text.length > 1000 ? '...' : ''}
            </div>
          </div>` : ''
        }
      </div>
    </div>
  `;
}

// Display AI analysis results
function displayAnalysisResults(data, mode) {
  const resultArea = document.getElementById('ia-result-text');
  if (!resultArea) return;
  
  if (mode === 'search') {
    const results = data.data?.results || [];
    if (results.length === 0) {
      resultArea.textContent = 'No se encontraron resultados';
      return;
    }
    
    resultArea.innerHTML = results
      .map(r => `
        <div class="mb-3 p-2 border rounded">
          <div class="d-flex justify-content-between">
            <strong>Similitud: ${(r.score * 100).toFixed(1)}%</strong>
            <small class="text-muted">${r.source || ''}</small>
          </div>
          <p class="mb-0 mt-1">${r.snippet || r.text || ''}</p>
        </div>
      `)
      .join('');
      
  } else if (mode === 'ner') {
    // Display NER results in a structured way
    const entities = data.data?.entities || [];
    if (entities.length === 0) {
      resultArea.textContent = 'No se encontraron entidades nombradas';
      return;
    }
    
    const grouped = entities.reduce((acc, { entity, type, score }) => {
      if (!acc[type]) acc[type] = [];
      acc[type].push({ entity, score });
      return acc;
    }, {});
    
    resultArea.innerHTML = Object.entries(grouped)
      .map(([type, items]) => `
        <div class="mb-3">
          <h6>${type}</h6>
          <div class="d-flex flex-wrap gap-1">
            ${items.map(item => 
              `<span class="badge bg-primary me-1 mb-1">
                ${item.entity} 
                <small>${(item.score * 100).toFixed(0)}%</small>
              </span>`
            ).join('')}
          </div>
        </div>
      `).join('');
  }
}

// Initialize dashboard when DOM is fully loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initDashboard);
} else {
  initDashboard();
}

// Update KPIs and Gantt chart
async function updateProjectVisuals() {
  const projectName = document.getElementById('project-name')?.value || 'PROY_1';
  
  try {
    // Update KPIs
    const kpiResponse = await fetch(`${window.AgenteIA.config.API_BASE}/api/projects/${encodeURIComponent(projectName)}/kpi`);
    const kpiData = await kpiResponse.json();
    
    const kpiContainer = document.getElementById('kpi-container');
    if (kpiContainer && kpiResponse.ok && kpiData.data) {
      kpiContainer.innerHTML = Object.entries(kpiData.data)
        .map(([kpi, value]) => `
          <div class="col-6 col-md-4 col-lg-3 mb-3">
            <div class="card h-100">
              <div class="card-body p-3 text-center">
                <div class="text-muted mb-1">${kpi}</div>
                <div class="h3 m-0">${value}</div>
              </div>
            </div>
          </div>
        `)
        .join('');
    }

    // Update Gantt chart
    const ganttResponse = await fetch(`${window.AgenteIA.config.API_BASE}/api/projects/${encodeURIComponent(projectName)}/gantt.png`);
    if (ganttResponse.ok) {
      const blob = await ganttResponse.blob();
      const url = URL.createObjectURL(blob);
      const ganttImg = document.getElementById('gantt-image');
      if (ganttImg) {
        ganttImg.src = url;
        ganttImg.onload = () => URL.revokeObjectURL(url);
      }
    }
    
    window.AgenteIA.utils.showAlert('Datos del proyecto actualizados correctamente', 'success');
  } catch (error) {
    console.error('Error updating project visuals:', error);
    window.AgenteIA.utils.showAlert('Error al actualizar los datos del proyecto', 'danger');
  }
}

// Add update function to global scope for manual triggering if needed
window.AgenteIA.updateProjectVisuals = updateProjectVisuals;

console.log('✅ dashboard.js cargado');
