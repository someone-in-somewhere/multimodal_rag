/**
 * static/app.js
 */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    API_BASE: window.location.origin,
    API_KEY: 'test-api-key',
    MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
    MAX_IMAGE_SIZE: 10 * 1024 * 1024, // 10MB for images
    TYPING_DELAY: 100, // ms between characters for typing effect
    AUTO_SCROLL_DELAY: 100,
    SUPPORTED_DOCUMENT_TYPES: [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/html',
        'text/plain',
        'text/markdown'
    ],
    SUPPORTED_IMAGE_TYPES: [
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp'
    ]
};

// ============================================================================
// State Management
// ============================================================================

const state = {
    uploadedDocs: [],
    currentImage: null,
    isUploading: false,
    isQuerying: false,
    messageHistory: []
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Application initialized');
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    addWelcomeMessage();
    
    // Load existing documents (if any)
    loadDocuments();
    
    // Check system health
    checkHealth();
    
    console.log('✅ App ready');
}

function setupEventListeners() {
    // Message input - Enter key
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keypress', handleKeyPress);
        messageInput.addEventListener('input', adjustTextareaHeight);
    }
    
    // Image input preview
    const imageInput = document.getElementById('imageInput');
    if (imageInput) {
        imageInput.addEventListener('change', previewImage);
    }
    
    // File input change
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileInputChange);
    }
    
    // Prevent form submission on Enter (allow Shift+Enter for new line)
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            sendMessage();
        });
    }
}

// ============================================================================
// Welcome Message
// ============================================================================

function addWelcomeMessage() {
    const welcomeMessages = [
        '👋 Xin chào! Tôi là trợ lý AI của bạn.',
        '📚 Hãy upload tài liệu và đặt câu hỏi cho tôi!',
        '💡 Tôi có thể giúp bạn tìm kiếm thông tin trong các tài liệu PDF, DOCX, và hình ảnh.'
    ];
    
    welcomeMessages.forEach((msg, index) => {
        setTimeout(() => {
            addBotMessage(msg, null, false);
        }, index * 500);
    });
}

// ============================================================================
// File Upload Management
// ============================================================================

function handleFileInputChange(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFiles();
    }
}

async function uploadFiles() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;
    
    if (files.length === 0) {
        showStatus('⚠️ Vui lòng chọn file!', 'error');
        return;
    }
    
    if (state.isUploading) {
        showStatus('⏳ Vui lòng đợi upload hiện tại hoàn thành', 'warning');
        return;
    }
    
    // Validate files
    const validFiles = Array.from(files).filter(file => {
        if (file.size > CONFIG.MAX_FILE_SIZE) {
            showStatus(`❌ File "${file.name}" quá lớn (max ${formatFileSize(CONFIG.MAX_FILE_SIZE)})`, 'error');
            return false;
        }
        return true;
    });
    
    if (validFiles.length === 0) {
        return;
    }
    
    state.isUploading = true;
    showStatus('⏳ Đang upload...', 'loading');
    
    let successCount = 0;
    let failCount = 0;
    
    for (const file of validFiles) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            console.log('📤 Uploading:', file.name);
            
            const response = await fetch(`${CONFIG.API_BASE}/upload`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${CONFIG.API_KEY}`
                },
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Upload success:', result);
                
                state.uploadedDocs.push({
                    id: result.doc_id,
                    name: result.filename,
                    chunks: result.chunks_processed
                });
                
                successCount++;
                
                // Add notification message
                addSystemMessage(`✅ Đã upload: ${file.name}`);
            } else {
                const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
                console.error('❌ Upload error:', error);
                showStatus(`❌ Lỗi upload "${file.name}": ${error.detail}`, 'error');
                failCount++;
            }
        } catch (error) {
            console.error('❌ Fetch error:', error);
            showStatus(`❌ Lỗi kết nối: ${error.message}`, 'error');
            failCount++;
        }
    }
    
    state.isUploading = false;
    fileInput.value = '';
    
    // Update UI
    updateDocumentsList();
    
    // Show final status
    if (failCount === 0) {
        showStatus(`✅ Upload thành công ${successCount} file!`, 'success');
    } else {
        showStatus(`⚠️ Upload ${successCount} thành công, ${failCount} thất bại`, 'warning');
    }
}

function updateDocumentsList() {
    const list = document.getElementById('documentsList');
    if (!list) return;
    
    if (state.uploadedDocs.length === 0) {
        list.innerHTML = '<li style="color: #999; font-style: italic;">Chưa có tài liệu nào</li>';
        return;
    }
    
    list.innerHTML = '';
    
    state.uploadedDocs.forEach((doc, index) => {
        const li = document.createElement('li');
        li.className = 'document-item';
        
        const totalChunks = (doc.chunks?.text || 0) + (doc.chunks?.table || 0) + (doc.chunks?.image || 0);
        
        li.innerHTML = `
            <div class="doc-info">
                <span class="doc-icon">📄</span>
                <span class="doc-name">${escapeHtml(doc.name)}</span>
            </div>
            <div class="doc-meta">
                <span title="Text chunks">📝 ${doc.chunks?.text || 0}</span>
                <span title="Tables">📊 ${doc.chunks?.table || 0}</span>
                <span title="Images">🖼️ ${doc.chunks?.image || 0}</span>
            </div>
        `;
        
        list.appendChild(li);
    });
}

async function loadDocuments() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/documents`, {
            headers: {
                'Authorization': `Bearer ${CONFIG.API_KEY}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.documents && data.documents.length > 0) {
                state.uploadedDocs = data.documents.map(doc => ({
                    id: doc.doc_id,
                    name: doc.filename,
                    chunks: doc.chunks
                }));
                updateDocumentsList();
                console.log(`📚 Loaded ${state.uploadedDocs.length} documents`);
            }
        }
    } catch (error) {
        console.error('❌ Failed to load documents:', error);
    }
}

// ============================================================================
// Image Preview
// ============================================================================

function previewImage() {
    const input = document.getElementById('imageInput');
    const preview = document.getElementById('imagePreview');
    
    if (!input.files || !input.files[0]) return;
    
    const file = input.files[0];
    
    // Validate file size
    if (file.size > CONFIG.MAX_IMAGE_SIZE) {
        showStatus(`❌ Ảnh quá lớn (max ${formatFileSize(CONFIG.MAX_IMAGE_SIZE)})`, 'error');
        input.value = '';
        return;
    }
    
    // Validate file type
    if (!CONFIG.SUPPORTED_IMAGE_TYPES.includes(file.type)) {
        showStatus('❌ Định dạng ảnh không được hỗ trợ', 'error');
        input.value = '';
        return;
    }
    
    const reader = new FileReader();
    
    reader.onload = (e) => {
        state.currentImage = e.target.result;
        
        preview.innerHTML = `
            <div class="image-preview-container">
                <img src="${state.currentImage}" alt="Preview">
                <button class="btn-clear-image" onclick="clearImage()" title="Xóa ảnh">
                    ❌
                </button>
            </div>
        `;
        
        preview.style.display = 'block';
        
        console.log('🖼️ Image preview loaded');
    };
    
    reader.onerror = () => {
        showStatus('❌ Lỗi đọc file ảnh', 'error');
        input.value = '';
    };
    
    reader.readAsDataURL(file);
}

function clearImage() {
    state.currentImage = null;
    
    const preview = document.getElementById('imagePreview');
    if (preview) {
        preview.innerHTML = '';
        preview.style.display = 'none';
    }
    
    const input = document.getElementById('imageInput');
    if (input) {
        input.value = '';
    }
    
    console.log('🗑️ Image cleared');
}

// ============================================================================
// Chat Interface
// ============================================================================

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    // Validate input
    if (!message && !state.currentImage) {
        showStatus('⚠️ Vui lòng nhập câu hỏi hoặc chọn ảnh', 'warning');
        return;
    }
    
    if (state.isQuerying) {
        showStatus('⏳ Vui lòng đợi câu trả lời hiện tại', 'warning');
        return;
    }
    
    // Check if documents are uploaded
    if (state.uploadedDocs.length === 0) {
        showStatus('⚠️ Vui lòng upload tài liệu trước!', 'warning');
        addBotMessage('📚 Bạn cần upload tài liệu trước khi đặt câu hỏi. Hãy sử dụng nút "Upload Tài liệu" bên trên!');
        return;
    }
    
    // Add user message to chat
    addUserMessage(message, state.currentImage);
    
    // Save to history
    state.messageHistory.push({
        role: 'user',
        content: message,
        image: state.currentImage,
        timestamp: new Date().toISOString()
    });
    
    // Clear input
    input.value = '';
    adjustTextareaHeight();
    
    // Clear image preview
    if (state.currentImage) {
        clearImage();
    }
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    state.isQuerying = true;
    
    try {
        console.log('💬 Sending query:', message);
        
        const response = await fetch(`${CONFIG.API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${CONFIG.API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message,
                top_k: 5,
                use_multimodal: state.currentImage !== null
            })
        });
        
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Query result:', result);
            
            // Add bot response with typing effect
            addBotMessage(result.answer, result.sources, true);
            
            // Save to history
            state.messageHistory.push({
                role: 'assistant',
                content: result.answer,
                sources: result.sources,
                timestamp: new Date().toISOString()
            });
            
            // Show processing time
            if (result.processing_time) {
                console.log(`⏱️ Processing time: ${result.processing_time.toFixed(2)}s`);
            }
        } else {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            console.error('❌ Query error:', error);
            
            addBotMessage(
                `❌ Xin lỗi, đã có lỗi xảy ra: ${error.detail}\n\nVui lòng thử lại!`,
                null,
                false
            );
        }
    } catch (error) {
        console.error('❌ Query fetch error:', error);
        removeTypingIndicator(typingId);
        
        addBotMessage(
            `❌ Lỗi kết nối đến máy chủ: ${error.message}\n\nVui lòng kiểm tra kết nối mạng và thử lại!`,
            null,
            false
        );
    } finally {
        state.isQuerying = false;
    }
}

function handleKeyPress(event) {
    // Send on Enter (without Shift)
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function adjustTextareaHeight() {
    const textarea = document.getElementById('messageInput');
    if (!textarea) return;
    
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// ============================================================================
// Message Display
// ============================================================================

function addUserMessage(text, image) {
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    
    let content = '';
    
    if (image) {
        content += `<img src="${image}" alt="User image" class="message-image">`;
    }
    
    if (text) {
        content += `<div class="message-text">${escapeHtml(text)}</div>`;
    }
    
    messageDiv.innerHTML = content;
    messagesDiv.appendChild(messageDiv);
    
    scrollToBottom();
}

function addBotMessage(text, sources = null, useTypingEffect = false) {
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    
    if (useTypingEffect) {
        // Add empty message first
        messageDiv.innerHTML = '<div class="message-text"></div>';
        messagesDiv.appendChild(messageDiv);
        scrollToBottom();
        
        // Type out the message
        typeMessage(messageDiv, text, sources);
    } else {
        // Add complete message immediately
        let html = `<div class="message-text">${formatMarkdown(text)}</div>`;
        
        if (sources && sources.length > 0) {
            html += formatSources(sources);
        }
        
        messageDiv.innerHTML = html;
        messagesDiv.appendChild(messageDiv);
        scrollToBottom();
    }
}

function addSystemMessage(text) {
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';
    messageDiv.innerHTML = `<div class="message-text">${escapeHtml(text)}</div>`;
    
    messagesDiv.appendChild(messageDiv);
    scrollToBottom();
}

async function typeMessage(messageDiv, text, sources) {
    const messageText = messageDiv.querySelector('.message-text');
    if (!messageText) return;
    
    const formattedText = formatMarkdown(text);
    
    // For typing effect, we'll add the complete formatted HTML
    // (A true character-by-character typing would require more complex parsing)
    messageText.innerHTML = formattedText;
    
    // Add sources if present
    if (sources && sources.length > 0) {
        const sourcesHtml = formatSources(sources);
        messageDiv.insertAdjacentHTML('beforeend', sourcesHtml);
    }
    
    scrollToBottom();
}

function formatSources(sources) {
    if (!sources || sources.length === 0) return '';
    
    const sourcesList = sources
        .map(s => {
            const relevance = (s.relevance_score * 100).toFixed(1);
            const icon = getTypeIcon(s.type);
            return `<span class="source-item">${icon} ${s.type} (${relevance}%)</span>`;
        })
        .join('');
    
    return `
        <div class="sources">
            <div class="sources-title">📚 Nguồn tham khảo:</div>
            <div class="sources-list">${sourcesList}</div>
        </div>
    `;
}

function getTypeIcon(type) {
    const icons = {
        'text': '📝',
        'table': '📊',
        'image': '🖼️'
    };
    return icons[type] || '📄';
}

// ============================================================================
// Typing Indicator
// ============================================================================

function addTypingIndicator() {
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv) return null;
    
    const typingDiv = document.createElement('div');
    const id = 'typing-' + Date.now();
    
    typingDiv.id = id;
    typingDiv.className = 'message bot typing-indicator';
    typingDiv.innerHTML = `
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    messagesDiv.appendChild(typingDiv);
    scrollToBottom();
    
    return id;
}

function removeTypingIndicator(id) {
    if (!id) return;
    
    const element = document.getElementById(id);
    if (element) {
        element.style.opacity = '0';
        setTimeout(() => element.remove(), 300);
    }
}

// ============================================================================
// Markdown Formatting
// ============================================================================

function formatMarkdown(text) {
    if (!text) return '';
    
    // Escape HTML first
    text = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Format markdown
    text = text
        // Headers
        .replace(/### (.*?)(\n|$)/g, '<h3>$1</h3>')
        .replace(/## (.*?)(\n|$)/g, '<h2>$1</h2>')
        .replace(/# (.*?)(\n|$)/g, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Code blocks
        .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
        // Inline code
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        // Lists
        .replace(/^\* (.+)$/gm, '<li>$1</li>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    
    // Wrap in paragraph if not already wrapped
    if (!text.startsWith('<')) {
        text = '<p>' + text + '</p>';
    }
    
    return text;
}

// ============================================================================
// Status Messages
// ============================================================================

function showStatus(message, type = 'info') {
    const status = document.getElementById('uploadStatus');
    if (!status) return;
    
    status.textContent = message;
    status.className = `status-message ${type}`;
    status.style.display = 'block';
    
    console.log(`${getStatusIcon(type)} ${message}`);
    
    // Auto-hide success messages
    if (type === 'success') {
        setTimeout(() => {
            status.style.display = 'none';
        }, 3000);
    }
}

function getStatusIcon(type) {
    const icons = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'loading': '⏳'
    };
    return icons[type] || 'ℹ️';
}

// ============================================================================
// Utility Functions
// ============================================================================

function scrollToBottom() {
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv) return;
    
    setTimeout(() => {
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }, CONFIG.AUTO_SCROLL_DELAY);
}

function escapeHtml(text) {
    if (!text) return '';
    
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function checkHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/health`);
        
        if (response.ok) {
            const data = await response.json();
            console.log('💚 System health:', data);
            
            if (data.status !== 'healthy') {
                showStatus('⚠️ Hệ thống đang gặp vấn đề', 'warning');
            }
        } else {
            console.warn('⚠️ Health check failed');
        }
    } catch (error) {
        console.error('❌ Health check error:', error);
    }
}

// ============================================================================
// Export functions for inline usage
// ============================================================================

window.uploadFiles = uploadFiles;
window.sendMessage = sendMessage;
window.previewImage = previewImage;
window.clearImage = clearImage;
window.handleKeyPress = handleKeyPress;

// ============================================================================
// Error Handling
// ============================================================================

window.addEventListener('error', (event) => {
    console.error('💥 Global error:', event.error);
    showStatus('Đã xảy ra lỗi không mong muốn', 'error');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('💥 Unhandled promise rejection:', event.reason);
});

console.log('📦 App.js loaded successfully');
