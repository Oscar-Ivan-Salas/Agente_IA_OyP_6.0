'use strict';

// Initialize AgenteIA.utils if not exists
if (!window.AgenteIA) window.AgenteIA = {};
if (!window.AgenteIA.utils) window.AgenteIA.utils = {};

// Add showAlert utility if not exists
if (!window.AgenteIA.utils.showAlert) {
  window.AgenteIA.utils.showAlert = function(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.getElementById('alerts-container') || document.body;
    container.prepend(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      const bsAlert = new bootstrap.Alert(alertDiv);
      bsAlert.close();
    }, 5000);
  };
}

// Toggle mock mode button
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('toggle-mock-btn');
  if (btn) {
    btn.addEventListener('click', () => {
      const current = localStorage.getItem('USE_MOCKS') !== '0';
      const newValue = current ? '0' : '1';
      localStorage.setItem('USE_MOCKS', newValue);
      const message = `Modo mocks ${newValue === '1' ? 'activado' : 'desactivado'}. Recargando...`;
      window.AgenteIA.utils.showAlert(message, 'info');
      setTimeout(() => location.reload(), 1000);
    });
  }
  
  // Log mock status
  const useMocks = localStorage.getItem('USE_MOCKS') !== '0';
  console.log(`🔧 Modo mocks: ${useMocks ? 'ACTIVADO' : 'DESACTIVADO'}`);
});

console.log('✅ model-manager.js cargado (control de mocks)');
