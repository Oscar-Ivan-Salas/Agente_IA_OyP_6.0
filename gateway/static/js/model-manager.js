'use strict';

document.addEventListener('DOMContentLoaded', function () {
  const AgenteIA = window.AgenteIA;
  if (!AgenteIA) {
    console.error('El objeto principal AgenteIA no está inicializado.');
    return;
  }

  const { showAlert } = AgenteIA.utils;

  class ModelManager {
    constructor() {
      this.elements = {
        // Elementos del selector de chat
        modelSelectorBtn: document.getElementById('model-selector'),
        selectedModelSpan: document.getElementById('selected-model'),
        modelStatusSpan: document.getElementById('model-status'),
        modelDropdown: document.getElementById('model-dropdown'),
        configureBtn: document.getElementById('configure-model-btn'),
        
        // Elementos del modal de configuración
        modalElement: document.getElementById('modelConfigModal'),
        modalTitle: document.getElementById('modelConfigModalLabel'),
        providerSelect: document.getElementById('model-provider'),
        apiKeyInput: document.getElementById('api-key'),
        toggleApiKeyBtn: document.getElementById('toggle-api-key'),
        temperatureSlider: document.getElementById('temperature'),
        temperatureValue: document.getElementById('temperature-value'),
        maxTokensSlider: document.getElementById('max-tokens'),
        maxTokensValue: document.getElementById('max-tokens-value'),
        testBtn: document.getElementById('test-model-btn'),
        saveBtn: document.getElementById('save-model-config'),
        testResultAlert: document.getElementById('model-test-result'),
      };
      this.elements.modal = this.elements.modalElement ? new bootstrap.Modal(this.elements.modalElement) : null;
      this.currentModel = null;
    }

    async init() {
      if (!this.elements.modelSelectorBtn) {
          console.log("No se encontraron elementos del gestor de modelos en esta página.");
          return;
      }
      await this.fetchModels();
      this.renderModelDropdown();
      this.bindEvents();
      if (AgenteIA.state.models.length > 0) {
        this.selectModel(AgenteIA.state.models[0].id);
      }
    }

    async fetchModels() {
      const mockModels = [
        { id: 'llama-3.1-8b', name: 'Llama 3.1 8B', provider: 'Local', status: 'disponible', params: { temp: 0.7, tokens: 2048 } },
        { id: 'mistral-7b', name: 'Mistral 7B', provider: 'Local', status: 'disponible', params: { temp: 0.8, tokens: 2048 } },
        { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI', status: 'requiere_api_key', params: { temp: 0.7, tokens: 4096 } },
        { id: 'claude-3', name: 'Claude 3 Sonnet', provider: 'Anthropic', status: 'requiere_api_key', params: { temp: 0.7, tokens: 4096 } },
      ];
      
      // En un caso real, esto sería una llamada a la API
      AgenteIA.state.models = mockModels;
      console.log('Modelos cargados:', AgenteIA.state.models);
    }

    renderModelDropdown() {
      const { models } = AgenteIA.state;
      if (!models || models.length === 0) {
        this.elements.modelDropdown.innerHTML = '<li><span class=\"dropdown-item-text\">No hay modelos disponibles.</span></li>';
        return;
      }

      const providers = [...new Set(models.map(m => m.provider))];
      let html = '';
      providers.forEach(provider => {
        html += `<li><h6 class=\"dropdown-header\">${provider}</h6></li>`;
        models.filter(m => m.provider === provider).forEach(model => {
          html += `<li><a class=\"dropdown-item\" href=\"#\" data-model-id=\"${model.id}\">${model.name}</a></li>`;
        });
      });
      this.elements.modelDropdown.innerHTML = html;
    }

    selectModel(modelId) {
      const model = AgenteIA.state.models.find(m => m.id === modelId);
      if (!model) return;

      this.currentModel = model;
      this.elements.selectedModelSpan.textContent = model.name;
      this.updateModelStatus();
      console.log(`Modelo seleccionado: ${model.name}`);
    }

    updateModelStatus() {
      const statusSpan = this.elements.modelStatusSpan;
      switch (this.currentModel.status) {
        case 'disponible':
          statusSpan.innerHTML = '<span class=\"badge bg-success-lt\">Disponible</span>';
          break;
        case 'requiere_api_key':
          statusSpan.innerHTML = '<span class=\"badge bg-warning-lt\">Configurar</span>';
          break;
        default:
          statusSpan.innerHTML = '';
      }
    }

    openConfigModal() {
      if (!this.currentModel) {
        showAlert('Por favor, selecciona un modelo primero.', 'warning');
        return;
      }
      if (!this.elements.modal) {
        console.error("El modal de configuración no está disponible.");
        return;
      }

      this.elements.modalTitle.textContent = `Configurar ${this.currentModel.name}`;
      this.elements.providerSelect.innerHTML = `<option>${this.currentModel.provider}</option>`;
      
      const config = this.loadConfig();
      this.elements.apiKeyInput.value = config.apiKey || '';
      this.elements.temperatureSlider.value = config.params.temp;
      this.elements.temperatureValue.textContent = config.params.temp;
      this.elements.maxTokensSlider.value = config.params.tokens;
      this.elements.maxTokensValue.textContent = config.params.tokens;

      const isCloud = this.currentModel.provider !== 'Local';
      this.elements.apiKeyInput.closest('.mb-3').style.display = isCloud ? 'block' : 'none';

      this.elements.modal.show();
    }

    loadConfig() {
      const defaultConfig = { apiKey: '', params: this.currentModel.params };
      try {
        const saved = localStorage.getItem(`model_config_${this.currentModel.id}`);
        return saved ? JSON.parse(saved) : defaultConfig;
      } catch (e) {
        return defaultConfig;
      }
    }

    saveConfig() {
      const config = {
        apiKey: this.elements.apiKeyInput.value,
        params: {
          temp: this.elements.temperatureSlider.value,
          tokens: this.elements.maxTokensSlider.value,
        }
      };

      try {
        localStorage.setItem(`model_config_${this.currentModel.id}`, JSON.stringify(config));
        showAlert('Configuración guardada con éxito.', 'success');
        this.elements.modal.hide();
      } catch (e) {
        showAlert('No se pudo guardar la configuración.', 'danger');
      }
    }

    bindEvents() {
      this.elements.modelDropdown.addEventListener('click', e => {
        if (e.target.matches('[data-model-id]')) {
          e.preventDefault();
          this.selectModel(e.target.dataset.modelId);
        }
      });

      this.elements.configureBtn.addEventListener('click', () => this.openConfigModal());
      this.elements.saveBtn.addEventListener('click', () => this.saveConfig());

      this.elements.temperatureSlider.addEventListener('input', e => {
        this.elements.temperatureValue.textContent = e.target.value;
      });

      this.elements.maxTokensSlider.addEventListener('input', e => {
        this.elements.maxTokensValue.textContent = e.target.value;
      });

      this.elements.toggleApiKeyBtn.addEventListener('click', () => {
        const input = this.elements.apiKeyInput;
        const icon = this.elements.toggleApiKeyBtn.querySelector('i');
        if (input.type === 'password') {
          input.type = 'text';
          icon.className = 'ti ti-eye-off';
        } else {
          input.type = 'password';
          icon.className = 'ti ti-eye';
        }
      });
    }
  }

  const modelManager = new ModelManager();
  modelManager.init();
  
  // Exponerlo globalmente si es necesario
  window.AgenteIA.modelManager = modelManager;
});
