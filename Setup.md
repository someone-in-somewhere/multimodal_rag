# Setup Guide - Multi-modal RAG System

## Prerequisites

1. **Python 3.9+**
2. **Redis** - For document storage
3. **Ollama** - For local LLM (recommended)
4. **Tesseract OCR** (optional) - For image text extraction

## Installation Steps

### 1. Clone Repository

\`\`\`bash
git clone <your-repo-url>
cd multimodal_rag
\`\`\`

### 2. Setup Python Environment

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 3. Install Redis

**Ubuntu/Debian:**
\`\`\`bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis
\`\`\`

**macOS:**
\`\`\`bash
brew install redis
brew services start redis
\`\`\`

**Windows:**
Download from: https://github.com/microsoftarchive/redis/releases

### 4. Install Ollama (for local LLM)

**Linux/Mac:**
\`\`\`bash
curl -fsSL https://ollama.com/install.sh | sh
\`\`\`

**Windows:**
Download from: https://ollama.com/download

**Pull model:**
\`\`\`bash
ollama pull gemma2:4b
# Or other models: mistral, llama3, phi3
\`\`\`

### 5. Configure Environment

\`\`\`bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
nano .env  # or your preferred editor
\`\`\`

**Important:** Change these values:
- `API_KEY` - Your secure API key
- `SECRET_KEY` - Random 32+ character string
- `OLLAMA_MODEL` - Model you pulled

### 6. Install Tesseract (Optional)

**Ubuntu/Debian:**
\`\`\`bash
sudo apt-get install tesseract-ocr
\`\`\`

**macOS:**
\`\`\`bash
brew install tesseract
\`\`\`

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

## Running the Application

### Start Services

1. **Start Redis:**
\`\`\`bash
redis-server
\`\`\`

2. **Start Ollama:**
\`\`\`bash
ollama serve
\`\`\`

3. **Start Application:**
\`\`\`bash
python run_server.py
\`\`\`

### Access Application

- **Chat Interface:** http://localhost:8000
- **Admin Panel:** http://localhost:8000/admin
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## Troubleshooting

### Redis Connection Error
\`\`\`bash
# Check if Redis is running
redis-cli ping
# Should return: PONG
\`\`\`

### Ollama Connection Error
\`\`\`bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
\`\`\`

### Model Not Found
\`\`\`bash
# List available models
ollama list

# Pull missing model
ollama pull gemma2:4b
\`\`\`

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup guide.
\`\`\`

Bạn có muốn tôi tạo thêm:
1. **README.md** - Project overview ✅
2. **DEPLOYMENT.md** - Production deployment guide ✅
3. **Docker files** - Containerization ✅
4. **Testing files** - Unit tests ✅

Gì tiếp theo? 😊
