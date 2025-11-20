/**
 * Admin Panel JavaScript for Multi-modal RAG System
 * Handles document upload, management, and UI interactions
 */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    API_BASE: window.location.origin,
    API_KEY: 'test-api-key',
    MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
    SUPPORTED_TYPES: [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/html',
        'text/plain',
        'text/markdown',
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp'
    ],
    UPLOAD_BATCH_SIZE: 3, // Number of concurrent uploads
    REFRESH_DELAY: 1000,
    TOAST_DURATION: 3000
};

// ============================================================================
// State Management
// ============================================================================

const state = {
    documents: [],
    deleteTarget: null,
    uploading: false,
    refreshing: false
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Admin panel initialized');
    initializeApp();
});

async function initializeApp() {
    try {
        setupEventListeners();
        setupDragDrop();
        await refreshDocuments();
        
        // Check API health
        await checkHealth();
        
        console.log('✅ App initialization complete');
    } catch (error) {
        console.error('❌ App initialization failed:', error);
        showToast('Lỗi khởi tạo ứng dụng', 'error');
    }
}

// ============================================================================
// Event Listeners Setup
// ============================================================================

function setupEventListeners() {
    // File input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileInputChange);
    }
    
    // Search input
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(filterDocuments, 300));
    }
    
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => refreshDocuments(true));
    }
    
    // Delete all button
    const deleteAllBtn = document.getElementById('deleteAllBtn');
    if (deleteAllBtn) {
        deleteAllBtn.addEventListener('click', deleteAllDocuments);
    }
    
    // Modal close on outside click
    const modal = document.getElementById('deleteModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeDeleteModal();
            }
        });
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

function handleKeyboardShortcuts(e) {
    // Escape to close modal
    if (e.key === 'Escape') {
        closeDeleteModal();
    }
    
    // Ctrl/Cmd + R to refresh
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        refreshDocuments(true);
    }
}

// ============================================================================
// Drag & Drop
// ============================================================================

function setupDragDrop() {
    const dropZone = document.getElementById('dropZone');
    if (!dropZone) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
    
    // Click to select files
    dropZone.addEventListener('click', () => {
        document.getElementById('fileInput').click();
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileInputChange(e) {
    handleFiles(e.target.files);
}

// ============================================================================
// File Upload
// ============================================================================

async function handleFiles(files) {
    if (!files || files.length === 0) return;
    
    if (state.uploading) {
        showToast('Vui lòng đợi upload hiện tại hoàn thành', 'warning');
        return;
    }
    
    const validFiles = Array.from(files).filter(file => {
        // Check file size
        if (file.size > CONFIG.MAX_FILE_SIZE) {
            showToast(
                `File "${file.name}" quá lớn (>${formatFileSize(CONFIG.MAX_FILE_SIZE)})`,
                'error'
            );
            return false;
        }
        
        // Check file type (optional - server will validate)
        // Uncomment to enable client-side type checking
        /*
        if (!CONFIG.SUPPORTED_TYPES.includes(file.type) && file.type !== '') {
            showToast(`File "${file.name}" không được hỗ trợ`, 'error');
            return false;
        }
        */
        
        return true;
    });
    
    if (validFiles.length === 0) {
        return;
    }
    
    state.uploading = true;
    const progressDiv = document.getElementById('uploadProgress');
    progressDiv.innerHTML = '';
    
    console.log(`📤 Uploading ${validFiles.length} file(s)`);
    
    // Upload files in batches
    const batches = chunkArray(validFiles, CONFIG.UPLOAD_BATCH_SIZE);
    
    for (const batch of batches) {
        await Promise.all(batch.map(file => uploadFile(file, progressDiv)));
    }
    
    state.uploading = false;
    
    // Clear file input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Refresh documents list
    setTimeout(() => refreshDocuments(), CONFIG.REFRESH_DELAY);
}

async function uploadFile(file, progressDiv) {
    const itemId = `upload-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    const itemDiv = document.createElement('div');
    itemDiv.className = 'upload-item';
    itemDiv.id = itemId;
    itemDiv.innerHTML = `
        <div class="upload-item-header">
            <div>
                <span class="upload-item-name">📄 ${escapeHtml(file.name)}</span>
                <span class="upload-item-size" style="color: #999; font-size: 0.85em; margin-left: 8px;">
                    (${formatFileSize(file.size)})
                </span>
            </div>
            <span class="upload-item-status">⏳ Đang upload...</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
        </div>
    `;
    progressDiv.appendChild(itemDiv);
    
    const progressFill = itemDiv.querySelector('.progress-fill');
    const statusSpan = itemDiv.querySelector('.upload-item-status');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        // Simulate progress (since we can't track real upload progress with fetch)
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 15;
                progress = Math.min(progress, 90);
                progressFill.style.width = progress + '%';
            }
        }, 200);
        
        const response = await fetch(`${CONFIG.API_BASE}/upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${CONFIG.API_KEY}`
            },
            body: formData
        });
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        
        if (response.ok) {
            const result = await response.json();
            statusSpan.innerHTML = '✅ Thành công';
            statusSpan.style.color = '#27ae60';
            
            console.log('✅ Upload success:', result);
            showToast(`Upload thành công: ${file.name}`, 'success');
            
            // Remove upload item after delay
            setTimeout(() => {
                itemDiv.style.opacity = '0';
                itemDiv.style.transform = 'translateX(20px)';
                setTimeout(() => itemDiv.remove(), 300);
            }, 2000);
        } else {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            statusSpan.innerHTML = '❌ Lỗi';
            statusSpan.style.color = '#e74c3c';
            
            console.error('❌ Upload failed:', error);
            showToast(`Lỗi upload "${file.name}": ${error.detail}`, 'error');
            
            // Keep failed items visible
            itemDiv.style.background = '#fff5f5';
        }
    } catch (error) {
        console.error('❌ Upload error:', error);
        
        statusSpan.innerHTML = '❌ Lỗi kết nối';
        statusSpan.style.color = '#e74c3c';
        showToast(`Lỗi kết nối: ${error.message}`, 'error');
        
        itemDiv.style.background = '#fff5f5';
    }
}

// ============================================================================
// Document Management
// ============================================================================

async function refreshDocuments(showLoading = false) {
    if (state.refreshing) return;
    
    state.refreshing = true;
    
    const tbody = document.getElementById('documentsBody');
    const emptyState = document.getElementById('emptyState');
    const refreshBtn = document.getElementById('refreshBtn');
    
    if (!tbody) {
        state.refreshing = false;
        return;
    }
    
    // Show loading state
    if (showLoading) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align:center; padding: 40px;">
                    <div class="spinner" style="width: 40px; height: 40px; margin: 0 auto 15px;"></div>
                    <div>Đang tải dữ liệu...</div>
                </td>
            </tr>
        `;
        
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<span class="spinner"></span> Đang tải...';
        }
    }
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/documents`, {
            headers: {
                'Authorization': `Bearer ${CONFIG.API_KEY}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            state.documents = data.documents || [];
            
            console.log(`📚 Loaded ${state.documents.length} documents`);
            
            if (state.documents.length === 0) {
                tbody.innerHTML = '';
                if (emptyState) {
                    emptyState.style.display = 'block';
                }
            } else {
                if (emptyState) {
                    emptyState.style.display = 'none';
                }
                renderDocumentsTable();
            }
            
            updateStats();
        } else {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            console.error('❌ Failed to fetch documents:', error);
            
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" style="text-align:center; padding: 40px; color: #e74c3c;">
                        ❌ Lỗi tải dữ liệu: ${error.detail}
                    </td>
                </tr>
            `;
            
            showToast('Lỗi tải danh sách tài liệu', 'error');
        }
    } catch (error) {
        console.error('❌ Network error:', error);
        
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align:center; padding: 40px; color: #e74c3c;">
                    ❌ Lỗi kết nối: ${error.message}
                </td>
            </tr>
        `;
        
        showToast('Lỗi kết nối máy chủ', 'error');
    } finally {
        state.refreshing = false;
        
        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.innerHTML = '🔄 Làm mới';
        }
    }
}

function renderDocumentsTable() {
    const tbody = document.getElementById('documentsBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    state.documents.forEach((doc, index) => {
        const row = tbody.insertRow();
        row.className = 'document-row';
        row.setAttribute('data-doc-id', doc.doc_id);
        
        const fileType = getFileType(doc.filename);
        const badgeClass = getTypeBadgeClass(fileType);
        
        row.innerHTML = `
            <td>
                <span style="color: #999; font-weight: 600;">#${index + 1}</span>
            </td>
            <td>
                <div class="doc-name">${escapeHtml(doc.filename || 'Untitled')}</div>
                <div class="doc-id">${escapeHtml(doc.doc_id)}</div>
            </td>
            <td>
                <span class="badge ${badgeClass}">${fileType}</span>
            </td>
            <td>
                <span style="font-weight: 600; color: #667eea;">
                    ${doc.chunks?.text || 0}
                </span>
            </td>
            <td>
                <span style="font-weight: 600; color: #2e7d32;">
                    ${doc.chunks?.table || 0}
                </span>
            </td>
            <td>
                <span style="font-weight: 600; color: #e65100;">
                    ${doc.chunks?.image || 0}
                </span>
            </td>
            <td>
                <span style="font-size: 0.9em; color: #666;">
                    ${formatDate(doc.timestamp)}
                </span>
            </td>
            <td>
                <div class="action-buttons">
                    <button 
                        class="btn-view" 
                        onclick="viewDocument('${escapeHtml(doc.doc_id)}')"
                        title="Xem chi tiết">
                        👁️ Xem
                    </button>
                    <button 
                        class="btn-delete" 
                        onclick="showDeleteModal('${escapeHtml(doc.doc_id)}')"
                        title="Xóa tài liệu">
                        🗑️ Xóa
                    </button>
                </div>
            </td>
        `;
    });
}

function updateStats() {
    let totalDocs = state.documents.length;
    let totalText = 0;
    let totalTables = 0;
    let totalImages = 0;
    
    state.documents.forEach(doc => {
        totalText += doc.chunks?.text || 0;
        totalTables += doc.chunks?.table || 0;
        totalImages += doc.chunks?.image || 0;
    });
    
    // Update stat cards with animation
    animateValue('totalDocs', totalDocs);
    animateValue('totalChunks', totalText);
    animateValue('totalTables', totalTables);
    animateValue('totalImages', totalImages);
}

function animateValue(elementId, targetValue) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const currentValue = parseInt(element.textContent) || 0;
    const duration = 500; // ms
    const steps = 20;
    const stepValue = (targetValue - currentValue) / steps;
    const stepDuration = duration / steps;
    
    let step = 0;
    const interval = setInterval(() => {
        step++;
        const newValue = Math.round(currentValue + (stepValue * step));
        element.textContent = newValue;
        
        if (step >= steps) {
            element.textContent = targetValue;
            clearInterval(interval);
        }
    }, stepDuration);
}

// ============================================================================
// Document Actions
// ============================================================================

function showDeleteModal(docId) {
    const doc = state.documents.find(d => d.doc_id === docId);
    if (!doc) return;
    
    state.deleteTarget = docId;
    
    const modal = document.getElementById('deleteModal');
    const messageEl = document.getElementById('deleteMessage');
    
    if (messageEl) {
        messageEl.innerHTML = `
            Bạn có chắc muốn xóa tài liệu này?<br><br>
            <strong>📄 ${escapeHtml(doc.filename || doc.doc_id)}</strong><br>
            <span style="font-size: 0.9em; color: #666;">
                (${doc.chunks?.text || 0} text, 
                ${doc.chunks?.table || 0} tables, 
                ${doc.chunks?.image || 0} images)
            </span>
        `;
    }
    
    if (modal) {
        modal.classList.add('show');
        
        // Focus on cancel button
        setTimeout(() => {
            const cancelBtn = modal.querySelector('.btn-secondary');
            if (cancelBtn) cancelBtn.focus();
        }, 100);
    }
}

function closeDeleteModal() {
    const modal = document.getElementById('deleteModal');
    if (modal) {
        modal.classList.remove('show');
    }
    state.deleteTarget = null;
}

async function confirmDelete() {
    if (!state.deleteTarget) return;
    
    const docId = state.deleteTarget;
    closeDeleteModal();
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/document/${docId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${CONFIG.API_KEY}`
            }
        });
        
        if (response.ok) {
            console.log('✅ Document deleted:', docId);
            showToast('Xóa tài liệu thành công!', 'success');
            
            // Remove from UI immediately
            const row = document.querySelector(`[data-doc-id="${docId}"]`);
            if (row) {
                row.style.opacity = '0';
                row.style.transform = 'translateX(-20px)';
                setTimeout(() => {
                    row.remove();
                    refreshDocuments();
                }, 300);
            } else {
                refreshDocuments();
            }
        } else {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            console.error('❌ Delete failed:', error);
            showToast(`Lỗi khi xóa: ${error.detail}`, 'error');
        }
    } catch (error) {
        console.error('❌ Delete error:', error);
        showToast(`Lỗi kết nối: ${error.message}`, 'error');
    }
}

async function deleteAllDocuments() {
    if (state.documents.length === 0) {
        showToast('Không có tài liệu nào để xóa', 'info');
        return;
    }
    
    const confirmMessage = `⚠️ XÓA TẤT CẢ ${state.documents.length} TÀI LIỆU?\n\nHành động này không thể hoàn tác!`;
    
    if (!confirm(confirmMessage)) return;
    
    const progressDiv = document.getElementById('uploadProgress');
    progressDiv.innerHTML = '<div style="text-align:center; padding:20px;">⏳ Đang xóa tất cả tài liệu...</div>';
    
    let deleted = 0;
    let failed = 0;
    
    for (const doc of state.documents) {
        try {
            const response = await fetch(`${CONFIG.API_BASE}/document/${doc.doc_id}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${CONFIG.API_KEY}`
                }
            });
            
            if (response.ok) {
                deleted++;
            } else {
                failed++;
            }
        } catch (error) {
            console.error('Delete error:', error);
            failed++;
        }
    }
    
    progressDiv.innerHTML = '';
    
    if (failed === 0) {
        showToast(`✅ Đã xóa thành công ${deleted} tài liệu!`, 'success');
    } else {
        showToast(`⚠️ Xóa ${deleted} thành công, ${failed} thất bại`, 'warning');
    }
    
    refreshDocuments();
}

function viewDocument(docId) {
    const doc = state.documents.find(d => d.doc_id === docId);
    if (!doc) {
        showToast('Không tìm thấy tài liệu', 'error');
        return;
    }
    
    const totalItems = (doc.chunks?.text || 0) + (doc.chunks?.table || 0) + (doc.chunks?.image || 0);
    
    const info = `
📄 THÔNG TIN TÀI LIỆU

📝 Tên file: ${doc.filename || 'N/A'}
🔑 Doc ID: ${doc.doc_id}

📊 Nội dung:
  • Text chunks: ${doc.chunks?.text || 0}
  • Tables: ${doc.chunks?.table || 0}
  • Images: ${doc.chunks?.image || 0}
  • Tổng items: ${totalItems}

⏰ Thời gian upload:
  ${formatDate(doc.timestamp)}
    `.trim();
    
    alert(info);
}

// ============================================================================
// Search & Filter
// ============================================================================

function filterDocuments() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase().trim();
    const rows = document.querySelectorAll('#documentsBody tr');
    
    let visibleCount = 0;
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        const isVisible = text.includes(searchTerm);
        
        row.style.display = isVisible ? '' : 'none';
        
        if (isVisible) {
            visibleCount++;
        }
    });
    
    console.log(`🔍 Search: "${searchTerm}" - ${visibleCount} results`);
}

// ============================================================================
// Health Check
// ============================================================================

async function checkHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/health`);
        
        if (response.ok) {
            const data = await response.json();
            console.log('💚 Health check:', data);
            
            if (data.status !== 'healthy') {
                showToast('⚠️ Hệ thống đang gặp vấn đề', 'warning');
            }
        } else {
            console.warn('⚠️ Health check failed');
        }
    } catch (error) {
        console.error('❌ Health check error:', error);
    }
}

// ============================================================================
// Toast Notifications
// ============================================================================

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span>${escapeHtml(message)}</span>
    `;
    
    toast.className = `toast ${type} show`;
    
    console.log(`${icons[type]} ${message}`);
    
    // Auto hide
    setTimeout(() => {
        toast.classList.remove('show');
    }, CONFIG.TOAST_DURATION);
}

// ============================================================================
// Utility Functions
// ============================================================================

function formatDate(isoString) {
    if (!isoString) return 'N/A';
    
    try {
        const date = new Date(isoString);
        
        // Check if valid date
        if (isNaN(date.getTime())) {
            return 'Invalid date';
        }
        
        return date.toLocaleString('vi-VN', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        return 'Invalid date';
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function getFileType(filename) {
    if (!filename) return 'DOC';
    
    const ext = filename.split('.').pop().toLowerCase();
    
    const types = {
        'pdf': 'PDF',
        'doc': 'DOCX',
        'docx': 'DOCX',
        'html': 'HTML',
        'htm': 'HTML',
        'txt': 'TEXT',
        'md': 'TEXT',
        'jpg': 'IMAGE',
        'jpeg': 'IMAGE',
        'png': 'IMAGE',
        'gif': 'IMAGE',
        'webp': 'IMAGE'
    };
    
    return types[ext] || 'DOC';
}

function getTypeBadgeClass(type) {
    const classes = {
        'PDF': 'badge-pdf',
        'DOCX': 'badge-docx',
        'HTML': 'badge-html',
        'TEXT': 'badge-text',
        'IMAGE': 'badge-image'
    };
    
    return classes[type] || 'badge-text';
}

function escapeHtml(text) {
    if (!text) return '';
    
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    
    return text.toString().replace(/[&<>"']/g, m => map[m]);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function chunkArray(array, size) {
    const chunks = [];
    for (let i = 0; i < array.length; i += size) {
        chunks.push(array.slice(i, i + size));
    }
    return chunks;
}

// ============================================================================
// Make functions globally accessible
// ============================================================================

window.refreshDocuments = refreshDocuments;
window.viewDocument = viewDocument;
window.showDeleteModal = showDeleteModal;
window.closeDeleteModal = closeDeleteModal;
window.confirmDelete = confirmDelete;
window.deleteAllDocuments = deleteAllDocuments;
window.filterDocuments = filterDocuments;

// ============================================================================
// Error Handling
// ============================================================================

window.addEventListener('error', (event) => {
    console.error('💥 Global error:', event.error);
    showToast('Đã xảy ra lỗi không mong muốn', 'error');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('💥 Unhandled promise rejection:', event.reason);
    showToast('Lỗi xử lý bất đồng bộ', 'error');
});

console.log('📦 Admin.js loaded successfully');
