const API_BASE = window.location.origin;
const API_KEY = 'test-api-key';

let uploadedDocs = [];
let currentImage = null;

document.addEventListener('DOMContentLoaded', () => {
    addBotMessage('👋 Xin chào! Tôi là trợ lý AI. Hãy upload tài liệu và đặt câu hỏi cho tôi!');
});

async function uploadFiles() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;
    
    if (files.length === 0) {
        showStatus('Vui lòng chọn file!', 'error');
        return;
    }
    s
    showStatus('Đang upload...', 'loading');
    
    for (let file of files) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            console.log('Uploading:', file.name);
            
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_KEY}`
                },
                body: formData
            });
            
            console.log('Response status:', response.status);
            
            if (response.ok) {
                const result = await response.json();
                console.log('Upload result:', result);
                uploadedDocs.push({
                    id: result.doc_id,
                    name: result.filename
                });
                updateDocumentsList();
                showStatus(`✅ Upload thành công: ${file.name}`, 'success');
            } else {
                const errorText = await response.text();
                console.error('Upload error:', errorText);
                showStatus(`❌ Lỗi upload: ${response.status}`, 'error');
            }
        } catch (error) {
            console.error('Fetch error:', error);
            showStatus(`❌ Lỗi: ${error.message}`, 'error');
        }
    }
    
    fileInput.value = '';
}

function updateDocumentsList() {
    const list = document.getElementById('documentsList');
    list.innerHTML = '';
    
    uploadedDocs.forEach(doc => {
        const li = document.createElement('li');
        li.textContent = `📄 ${doc.name}`;
        list.appendChild(li);
    });
}

function showStatus(message, type) {
    const status = document.getElementById('uploadStatus');
    status.textContent = message;
    status.className = `status-message ${type}`;
    
    if (type === 'success') {
        setTimeout(() => {
            status.textContent = '';
            status.className = 'status-message';
        }, 3000);
    }
}

function previewImage() {
    const input = document.getElementById('imageInput');
    const preview = document.getElementById('imagePreview');
    
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImage = e.target.result;
            preview.innerHTML = `<img src="${currentImage}" alt="Preview"><button onclick="clearImage()">❌</button>`;
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function clearImage() {
    currentImage = null;
    document.getElementById('imagePreview').innerHTML = '';
    document.getElementById('imageInput').value = '';
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message && !currentImage) return;
    
    addUserMessage(message, currentImage);
    
    input.value = '';
    document.getElementById('imagePreview').innerHTML = '';
    currentImage = null;
    
    const typingId = addTypingIndicator();
    
    try {
        console.log('Sending query:', message);
        
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message,
                top_k: 5,
                use_multimodal: false
            })
        });
        
        removeTypingIndicator(typingId);
        
        console.log('Query response status:', response.status);
        
        if (response.ok) {
            const result = await response.json();
            console.log('Query result:', result);
            addBotMessage(result.answer, result.sources);
        } else {
            const errorText = await response.text();
            console.error('Query error:', errorText);
            addBotMessage(`❌ Lỗi: ${response.status}`);
        }
    } catch (error) {
        console.error('Query fetch error:', error);
        removeTypingIndicator(typingId);
        addBotMessage(`❌ Lỗi kết nối: ${error.message}`);
    }
}

function addUserMessage(text, image) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    
    let content = '';
    if (image) {
        content += `<img src="${image}" alt="User image">`;
    }
    if (text) {
        content += `<div>${escapeHtml(text)}</div>`;
    }
    
    messageDiv.innerHTML = content;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addBotMessage(text, sources = null) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    
    let html = formatMarkdown(text);
    
    if (sources && sources.length > 0) {
        html += '<div class="sources">📚 Nguồn: ' + 
                sources.map(s => `${s.rank}. ${s.type} (${(s.relevance_score * 100).toFixed(1)}%)`).join(', ') +
                '</div>';
    }
    
    messageDiv.innerHTML = html;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function formatMarkdown(text) {
    return text
        .replace(/### (.*?)(\n|$)/g, '<h3>$1</h3>')
        .replace(/## (.*?)(\n|$)/g, '<h2>$1</h2>')
        .replace(/# (.*?)(\n|$)/g, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function addTypingIndicator() {
    const messagesDiv = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    const id = 'typing-' + Date.now();
    typingDiv.id = id;
    typingDiv.className = 'message bot typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}
