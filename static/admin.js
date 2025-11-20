
const API_BASE = window.location.origin;
const API_KEY = 'test-api-key';

let documents = [];
let deleteTarget = null;

document.addEventListener('DOMContentLoaded', () => {
    setupDragDrop();
    setupFileInput();
    refreshDocuments();
});

function setupDragDrop() {
    const dropZone = document.getElementById('dropZone');
    
    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });
    
    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });
    
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        handleFiles(files);
    });
}

function setupFileInput() {
    document.getElementById('fileInput').addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

async function handleFiles(files) {
    if (files.length === 0) return;
    
    const progressDiv = document.getElementById('uploadProgress');
    progressDiv.innerHTML = '';
    
    for (let file of files) {
        if (file.size > 10 * 1024 * 1024) {
            showToast(`File ${file.name} quá lớn (>10MB)`, 'error');
            continue;
        }
        await uploadFile(file, progressDiv);
    }
    
    document.getElementById('fileInput').value = '';
    setTimeout(() => refreshDocuments(), 1000);
}

async function uploadFile(file, progressDiv) {
    const itemDiv = document.createElement('div');
    itemDiv.className = 'upload-item';
    itemDiv.innerHTML = `
        <div class="upload-item-header">
            <span>📄 ${file.name}</span>
            <span class="status">Đang upload...</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
        </div>
    `;
    progressDiv.appendChild(itemDiv);
    
    const progressFill = itemDiv.querySelector('.progress-fill');
    const statusSpan = itemDiv.querySelector('.status');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += 10;
                progressFill.style.width = progress + '%';
            }
        }, 200);
        
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${API_KEY}` },
            body: formData
        });
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        
        if (response.ok) {
            const result = await response.json();
            statusSpan.textContent = '✅ Thành công';
            statusSpan.style.color = '#27ae60';
            showToast(`Upload thành công: ${file.name}`, 'success');
            setTimeout(() => itemDiv.remove(), 2000);
        } else {
            const error = await response.json();
            statusSpan.textContent = '❌ Lỗi';
            statusSpan.style.color = '#e74c3c';
            showToast(`Lỗi: ${error.detail}`, 'error');
        }
    } catch (error) {
        statusSpan.textContent = '❌ Lỗi kết nối';
        statusSpan.style.color = '#e74c3c';
        showToast(`Lỗi: ${error.message}`, 'error');
    }
}

async function refreshDocuments() {
    const tbody = document.getElementById('documentsBody');
    const emptyState = document.getElementById('emptyState');
    
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center">Đang tải...</td></tr>';
    
    try {
        const response = await fetch(`${API_BASE}/documents`, {
            headers: { 'Authorization': `Bearer ${API_KEY}` }
        });
        
        if (response.ok) {
            const data = await response.json();
            documents = data.documents || [];
            
            if (documents.length === 0) {
                tbody.innerHTML = '';
                emptyState.style.display = 'block';
                updateStats();
                return;
            }
            
            emptyState.style.display = 'none';
            tbody.innerHTML = '';
            
            documents.forEach((doc, index) => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>#${index + 1}</td>
                    <td><code>${doc.doc_id.substring(0, 12)}...</code></td>
                    <td><span class="badge badge-text">DOC</span></td>
                    <td>${doc.chunks?.text || 0}</td>
                    <td>${doc.chunks?.table || 0}</td>
                    <td>${doc.chunks?.image || 0}</td>
                    <td>${formatDate(doc.timestamp)}</td>
                    <td>
                        <div class="action-buttons">
                            <button class="btn-view" onclick="viewDocument('${doc.doc_id}')">👁️</button>
                            <button class="btn-delete" onclick="showDeleteModal('${doc.doc_id}')">🗑️</button>
                        </div>
                    </td>
                `;
            });
            
            updateStats();
        } else {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color: red;">Lỗi tải dữ liệu</td></tr>';
        }
    } catch (error) {
        console.error('Error:', error);
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color: red;">Lỗi kết nối</td></tr>';
    }
}

function updateStats() {
    let totalDocs = documents.length;
    let totalText = 0, totalTables = 0, totalImages = 0;
    
    documents.forEach(doc => {
        totalText += doc.chunks?.text || 0;
        totalTables += doc.chunks?.table || 0;
        totalImages += doc.chunks?.image || 0;
    });
    
    document.getElementById('totalDocs').textContent = totalDocs;
    document.getElementById('totalChunks').textContent = totalText;
    document.getElementById('totalImages').textContent = totalImages;
    document.getElementById('totalTables').textContent = totalTables;
}

function showDeleteModal(docId) {
    deleteTarget = docId;
    const modal = document.getElementById('deleteModal');
    document.getElementById('deleteMessage').textContent = `Bạn có chắc muốn xóa tài liệu này?`;
    modal.classList.add('show');
}

function closeDeleteModal() {
    document.getElementById('deleteModal').classList.remove('show');
    deleteTarget = null;
}

async function confirmDelete() {
    if (!deleteTarget) return;
    
    try {
        const response = await fetch(`${API_BASE}/document/${deleteTarget}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${API_KEY}` }
        });
        
        if (response.ok) {
            showToast('Xóa thành công!', 'success');
            refreshDocuments();
        } else {
            showToast('Lỗi khi xóa!', 'error');
        }
    } catch (error) {
        showToast(`Lỗi: ${error.message}`, 'error');
    }
    
    closeDeleteModal();
}

async function deleteAllDocuments() {
    if (!confirm('⚠️ XÓA TẤT CẢ TÀI LIỆU?\n\nHành động này không thể hoàn tác!')) return;
    
    for (let doc of documents) {
        try {
            await fetch(`${API_BASE}/document/${doc.doc_id}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${API_KEY}` }
            });
        } catch (error) {
            console.error('Delete error:', error);
        }
    }
    
    showToast('Đã xóa tất cả!', 'success');
    refreshDocuments();
}

function viewDocument(docId) {
    const doc = documents.find(d => d.doc_id === docId);
    if (!doc) return;
    
    const info = `
📄 Thông Tin Tài Liệu

Doc ID: ${doc.doc_id}
Text Chunks: ${doc.chunks?.text || 0}
Tables: ${doc.chunks?.table || 0}
Images: ${doc.chunks?.image || 0}
Total Items: ${doc.item_count || 0}
Thời gian: ${formatDate(doc.timestamp)}
    `.trim();
    
    alert(info);
}

function filterDocuments() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const rows = document.querySelectorAll('#documentsBody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(searchTerm) ? '' : 'none';
    });
}

function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('vi-VN', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}
