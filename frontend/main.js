import { marked } from 'marked';

// --- State Variables ---
let selectedFiles = [];
let simplifiedTextResult = '';
let currentSessionUuid = '';

// --- Configure Marked options ---
marked.setOptions({
  breaks: true,
  gfm: true
});

// --- DOM Elements ---
const apiStatusBadge = document.getElementById('api-status');
const toggleSettingsBtn = document.getElementById('toggle-settings-btn');
const settingsBody = document.getElementById('settings-body');
const modelSelect = document.getElementById('model-select');
const uuidInput = document.getElementById('uuid-input');
const regenerateUuidBtn = document.getElementById('regenerate-uuid-btn');

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const filesListContainer = document.getElementById('files-list-container');
const filesCountSpan = document.getElementById('files-count');
const filesList = document.getElementById('files-list');
const clearFilesBtn = document.getElementById('clear-files-btn');

const submitBtn = document.getElementById('submit-btn');
const submitLoader = document.getElementById('submit-loader');

const emptyState = document.getElementById('empty-state');
const loadingSkeleton = document.getElementById('loading-skeleton');
const outputViewer = document.getElementById('output-viewer');
const outputContent = document.getElementById('output-content');
const resultActions = document.getElementById('result-actions');

const copyBtn = document.getElementById('copy-btn');
const downloadPdfBtn = document.getElementById('download-pdf-btn');
const clearOutputBtn = document.getElementById('clear-output-btn');
const toastContainer = document.getElementById('toast-container');

// --- Toast System ---
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  
  let iconClass = 'fa-circle-info';
  if (type === 'success') iconClass = 'fa-circle-check';
  if (type === 'warning') iconClass = 'fa-triangle-exclamation';
  if (type === 'error') iconClass = 'fa-circle-xmark';
  
  toast.innerHTML = `
    <i class="fa-solid ${iconClass} toast-icon"></i>
    <div class="toast-body">${message}</div>
    <button class="toast-close"><i class="fa-solid fa-xmark"></i></button>
  `;
  
  toastContainer.appendChild(toast);
  
  toast.querySelector('.toast-close').addEventListener('click', () => {
    removeToast(toast);
  });
  
  setTimeout(() => {
    removeToast(toast);
  }, 4000);
}

function removeToast(toast) {
  toast.classList.add('removing');
  toast.addEventListener('animationend', () => {
    toast.remove();
  });
}

// --- Check API Connection ---
async function checkApiConnection() {
  try {
    const response = await fetch('/api/v1/upload', {
      method: 'OPTIONS'
    }).catch(() => {
      return fetch('/docs', { method: 'HEAD' });
    });
    
    if (response.ok || response.status === 405) {
      apiStatusBadge.className = 'status-pill ok';
      apiStatusBadge.querySelector('.status-text').textContent = 'API Connected';
    } else {
      throw new Error();
    }
  } catch (error) {
    apiStatusBadge.className = 'status-pill err';
    apiStatusBadge.querySelector('.status-text').textContent = 'API Disconnected';
    showToast('Failed to connect to the FastAPI backend service.', 'error');
  }
}

// --- Settings Toggle ---
toggleSettingsBtn.addEventListener('click', () => {
  settingsBody.classList.toggle('collapsed');
  toggleSettingsBtn.classList.toggle('flipped');
});

// Generate Random Session UUID
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

regenerateUuidBtn.addEventListener('click', () => {
  const newUuid = generateUUID();
  uuidInput.value = newUuid;
  showToast(`Generated new Session UUID: ${newUuid.substring(0, 8)}...`, 'info');
});

// --- Upload Zone Logic ---
browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropzone.addEventListener('click', () => {
  fileInput.click();
});

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('over');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('over');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('over');
  if (e.dataTransfer.files.length > 0) {
    handleFilesAdded(e.dataTransfer.files);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    handleFilesAdded(fileInput.files);
  }
});

function handleFilesAdded(files) {
  const validExtensions = ['.pdf', '.docx', '.txt'];
  const maxBytes = 20 * 1024 * 1024; // 20MB
  
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(extension)) {
      showToast(`Invalid file type for: ${file.name}. Only PDF, DOCX, TXT allowed.`, 'warning');
      continue;
    }
    
    if (file.size > maxBytes) {
      showToast(`File ${file.name} is too large. Limit is 20MB.`, 'warning');
      continue;
    }
    
    if (selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
      continue;
    }
    
    selectedFiles.push(file);
  }
  
  updateFilesListUI();
  updateSubmitButtonState();
}

function updateFilesListUI() {
  if (selectedFiles.length === 0) {
    filesListContainer.style.display = 'none';
    return;
  }
  
  filesListContainer.style.display = 'block';
  filesCountSpan.textContent = selectedFiles.length;
  filesList.innerHTML = '';
  
  selectedFiles.forEach((file, index) => {
    const li = document.createElement('li');
    li.className = 'file-row';
    
    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeStr = sizeKB > 1000 ? `${(sizeKB / 1024).toFixed(1)} MB` : `${sizeKB} KB`;
    
    let fileIcon = 'fa-file-lines';
    if (file.name.endsWith('.pdf')) fileIcon = 'fa-file-pdf';
    else if (file.name.endsWith('.docx')) fileIcon = 'fa-file-word';
    
    li.innerHTML = `
      <div class="file-meta">
        <i class="fa-regular ${fileIcon}"></i>
        <span class="file-name" title="${file.name}">${file.name}</span>
        <span class="file-size">${sizeStr}</span>
      </div>
      <button class="file-del" data-index="${index}"><i class="fa-solid fa-trash"></i></button>
    `;
    
    li.querySelector('.file-del').addEventListener('click', (e) => {
      e.stopPropagation();
      selectedFiles.splice(index, 1);
      updateFilesListUI();
      updateSubmitButtonState();
    });
    
    filesList.appendChild(li);
  });
}

clearFilesBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  selectedFiles = [];
  updateFilesListUI();
  updateSubmitButtonState();
  showToast('Cleared all selected files.', 'info');
});

// --- Submit Activation Logic ---
function updateSubmitButtonState() {
  submitBtn.disabled = selectedFiles.length === 0;
}

// --- Submit Request to API ---
submitBtn.addEventListener('click', async () => {
  const modelName = modelSelect.value;
  let sessionUuid = uuidInput.value.trim();
  
  if (!sessionUuid) {
    sessionUuid = generateUUID();
    uuidInput.value = sessionUuid;
  }
  
  currentSessionUuid = sessionUuid;
  setProcessingState(true);
  
  try {
    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });
    formData.append('uuid', sessionUuid);
    formData.append('response_format', 'json');
    if (modelName) {
      formData.append('model', modelName);
    }
    
    const response = await fetch('/api/v1/upload', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'API request processing failed.' }));
      throw new Error(errorData.detail || 'Simplification failed.');
    }
    
    const result = await response.json();
    simplifiedTextResult = result.simplified_content;
    
    renderSimplifiedOutput(simplifiedTextResult);
    showToast('Document simplified successfully!', 'success');
    
  } catch (error) {
    console.error('Simplification Error:', error);
    showToast(error.message || 'An error occurred during simplification.', 'error');
    setProcessingState(false, true);
  } finally {
    setProcessingState(false);
  }
});

function setProcessingState(isLoading, isError = false) {
  if (isLoading) {
    submitBtn.disabled = true;
    submitLoader.style.display = 'inline-block';
    submitBtn.querySelector('.btn-text').textContent = 'Processing...';
    
    emptyState.style.display = 'none';
    outputViewer.style.display = 'none';
    resultActions.style.display = 'none';
    loadingSkeleton.style.display = 'flex';
  } else {
    submitLoader.style.display = 'none';
    submitBtn.querySelector('.btn-text').textContent = 'Simplify Document';
    updateSubmitButtonState();
    loadingSkeleton.style.display = 'none';
    
    if (isError) {
      emptyState.style.display = 'flex';
      outputViewer.style.display = 'none';
      resultActions.style.display = 'none';
    }
  }
}

function renderSimplifiedOutput(markdownText) {
  if (!markdownText) {
    emptyState.style.display = 'flex';
    outputViewer.style.display = 'none';
    resultActions.style.display = 'none';
    return;
  }
  
  outputContent.innerHTML = marked.parse(markdownText);
  emptyState.style.display = 'none';
  outputViewer.style.display = 'block';
  resultActions.style.display = 'flex';
  
  if (window.innerWidth <= 1024) {
    outputViewer.scrollIntoView({ behavior: 'smooth' });
  }
}

// --- Action Buttons Actions ---

// Copy Text
copyBtn.addEventListener('click', async () => {
  if (!simplifiedTextResult) return;
  
  try {
    await navigator.clipboard.writeText(simplifiedTextResult);
    
    const originalText = copyBtn.innerHTML;
    copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
    copyBtn.classList.add('btn-success');
    showToast('Copied simplified text to clipboard.', 'success');
    
    setTimeout(() => {
      copyBtn.innerHTML = originalText;
      copyBtn.classList.remove('btn-success');
    }, 2000);
  } catch (err) {
    showToast('Failed to copy text to clipboard.', 'error');
  }
});

// Download PDF
downloadPdfBtn.addEventListener('click', async () => {
  if (!simplifiedTextResult) return;
  
  downloadPdfBtn.disabled = true;
  const originalHtml = downloadPdfBtn.innerHTML;
  downloadPdfBtn.innerHTML = '<span class="loader"></span> <span>Generating...</span>';
  
  try {
    const formData = new FormData();
    formData.append('content', simplifiedTextResult);
    formData.append('title', 'Simplified Compliance Summary');
    
    const response = await fetch('/api/v1/generate-pdf', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) throw new Error('PDF Generation Endpoint Failed.');
    
    const blob = await response.blob();
    const downloadUrl = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `simplified_compliance_${currentSessionUuid.substring(0, 8)}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    
    showToast('PDF summary downloaded successfully.', 'success');
  } catch (error) {
    console.error('PDF Download error:', error);
    showToast('Failed to download PDF summary.', 'error');
  } finally {
    downloadPdfBtn.disabled = false;
    downloadPdfBtn.innerHTML = originalHtml;
  }
});

// Clear Output & Reset
clearOutputBtn.addEventListener('click', () => {
  simplifiedTextResult = '';
  renderSimplifiedOutput('');
  showToast('Reset result view.', 'info');
});

// Run API connectivity check on load
checkApiConnection();
