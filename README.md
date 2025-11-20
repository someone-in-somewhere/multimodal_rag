# 🤖 Multi-modal RAG System

A production-ready FastAPI-based Retrieval-Augmented Generation (RAG) system that supports **text, tables, and images**.

## ✨ Features

- 📄 **Multi-format support**: PDF, DOCX, HTML, TXT, Markdown, Images
- 🔍 **Semantic search**: Vector-based similarity search with ChromaDB
- 🧠 **LLM backends**: Supports both Ollama (local) and OpenAI (cloud)
- 🖼️ **Multimodal**: Process and query images alongside text
- 📊 **Table extraction**: Parse and understand tabular data
- ⚡ **Fast & efficient**: Async processing, batch operations
- 🔐 **Secure**: API key authentication
- 📱 **Modern UI**: Responsive web interface with admin panel

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Chat + Admin Panel)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Server                          │
│  /upload  │  /query  │  /documents  │  /health          │
└─────┬───────┬────────┬─────────┬─────────────────────────┘
      │       │        │         │
      ▼       ▼        ▼         ▼
┌─────────┐ ┌──────┐ ┌────────┐ ┌─────────┐
│ Parser  │ │Embed │ │Retrieve│ │   LLM   │
└─────────┘ └──────┘ └────────┘ └─────────┘
      │         │         │          │
      ▼         ▼         ▼          ▼
┌─────────┐ ┌──────┐ ┌────────┐ ┌─────────┐
│ Files   │ │Chrome│ │ Redis  │ │ Ollama  │
└─────────┘ └──────┘ └────────┘ └─────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Redis
- Ollama (for local LLM) or OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd multimodal_rag
```

2. **Create virtual environment**
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup Redis**
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Or use Docker
docker run -d -p 6379:6379 redis:alpine
```

5. **Setup Ollama (for local LLM)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull gemma2:4b
```

6. **Configure environment**
```bash
cp .env.example .env
nano .env  # Edit with your settings
```

**Important settings:**
- `API_KEY`: Your API key for authentication
- `SECRET_KEY`: Random 32+ character string
- `OLLAMA_MODEL`: Model you pulled (e.g., `gemma2:4b`)
- `USE_LOCAL_LLM`: `true` for Ollama, `false` for OpenAI

7. **Run the server**
```bash
python run_server.py
```

8. **Access the application**

- **Chat Interface**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/admin
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage

### Upload Documents

1. Go to the **Admin Panel** (http://localhost:8000/admin)
2. Click "Upload Documents"
3. Select files (PDF, DOCX, TXT, Images, etc.)
4. Wait for processing

### Ask Questions

1. Go to the **Chat Interface** (http://localhost:8000)
2. Type your question
3. Get AI-powered answers based on your documents

### API Usage

**Upload Document:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Authorization: Bearer test-api-key" \
  -F "file=@document.pdf"
```

**Query Documents:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the sales figures?",
    "top_k": 5,
    "use_multimodal": true
  }'
```

## 🔧 Configuration

### Environment Variables

Key configurations in `.env`:
```bash
# LLM Backend
USE_LOCAL_LLM=true              # true for Ollama, false for OpenAI
OLLAMA_MODEL=gemma2:4b          # Ollama model
OPENAI_API_KEY=sk-...           # OpenAI API key

# Security
API_KEY=your-api-key            # API authentication
SECRET_KEY=your-secret-key      # JWT secret (32+ chars)

# Database
REDIS_HOST=localhost
REDIS_PORT=6379
CHROMA_PERSIST_DIR=./chroma_db

# Processing
CHUNK_SIZE=1000                 # Text chunk size
CHUNK_OVERLAP=200               # Chunk overlap
TOP_K_RESULTS=5                 # Number of results
```

### Supported Models

**Ollama (Local):**
- gemma2:4b (recommended)
- mistral
- llama3
- phi3
- qwen

**OpenAI (Cloud):**
- gpt-4o (multimodal)
- gpt-4-turbo
- gpt-3.5-turbo

## 🧪 Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/

# With coverage
pytest --cov=app tests/
```

## 📚 Project Structure
```
multimodal_rag/
├── app/
│   ├── models/          # LLM adapters
│   ├── server/          # FastAPI routes & auth
│   └── utils/           # Document processing, embedding, retrieval
├── static/              # Frontend assets (CSS, JS)
├── templates/           # HTML templates
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── run_server.py       # Server launcher
└── .env                # Environment variables
```

## 🐛 Troubleshooting

### Redis Connection Error
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Start Redis if not running
sudo systemctl start redis
```

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Model Not Found
```bash
# List installed models
ollama list

# Pull missing model
ollama pull gemma2:4b
```

### Port Already in Use
```bash
# Change port in .env
API_PORT=8001

# Or kill process using port 8000
lsof -ti:8000 | xargs kill -9
```

## 🚀 Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup guide.

Key considerations:
- Use strong `SECRET_KEY` and `API_KEY`
- Set `USE_LOCAL_LLM=false` if using OpenAI
- Configure CORS properly
- Use environment-specific `.env` files
- Set up SSL/TLS
- Use process manager (systemd, supervisor)
- Configure monitoring

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📧 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: your-email@example.com

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.ai/)
- [OpenAI](https://openai.com/)

---

Made with ❤️ by Multi-modal RAG Team
