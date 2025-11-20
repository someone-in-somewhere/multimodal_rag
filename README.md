
# Multi-modal RAG System

A production-ready Multi-modal Retrieval-Augmented Generation (RAG) system that supports text, tables, and images with semantic summarization and flexible LLM/MLLM backends.

## Features

- 📄 **Multi-format Support**: PDF, DOCX, HTML, and images
- 🔍 **Semantic Search**: Uses embeddings for accurate retrieval
- 🧠 **Dual Model Support**: 
  - LLM for text-only queries (GPT-4, Vicuna, Mistral)
  - MLLM for multimodal queries (GPT-4o, MiniGPT-4, LLaVA)
- 📊 **Table Processing**: Extracts and converts tables to Markdown
- 🖼️ **Image Understanding**: Processes and analyzes images
- 🔐 **Authentication**: API key and JWT token support
- ⚡ **Async API**: FastAPI with high performance
- 📦 **Vector Storage**: ChromaDB for embeddings
- 💾 **Document Store**: Redis for raw content

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         FastAPI Server              │
├─────────────────────────────────────┤
│  /upload  │  /query  │  /delete    │
└──────┬──────────┬──────────┬────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌─────────┐ ┌──────────┐
│  Parser  │ │ Embedder│ │Retriever │
└────┬─────┘ └────┬────┘ └────┬─────┘
     │            │            │
     ▼            ▼            ▼
┌─────────────────────────────────────┐
│  Summarizer (MLLM/LLM)              │
└─────────────────────────────────────┘
     │            │            │
     ▼            ▼            ▼
┌──────────┐ ┌─────────┐ ┌──────────┐
│ChromaDB  │ │ Redis   │ │  Files   │
│(Vectors) │ │(Docstore│ │(./figures)│
└──────────┘ └─────────┘ └──────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- Redis server
- OpenAI API key (or local LLM setup)
- Tesseract OCR (optional, for OCR)

### Step 1: Clone and Setup

```bash
# Create project directory
mkdir multimodal_rag
cd multimodal_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils redis-server

# macOS
brew install tesseract poppler redis

# Start Redis
redis-server
```

### Step 3: Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key-here
API_KEY=your-secure-api-key
SECRET_KEY=your-secret-key-for-jwt
```

### Step 4: Run the Server

```bash
python run_server.py
```

The server will start on `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## Usage

### 1. Upload Documents

```python
import requests

url = "http://localhost:8000/upload"
headers = {"Authorization": "Bearer your-api-key"}

with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files, headers=headers)

print(response.json())
# {
#   "doc_id": "doc_abc123",
#   "filename": "document.pdf",
#   "chunks_processed": {"text": 10, "table": 2, "image": 3},
#   "message": "Document processed successfully"
# }
```

### 2. Query System

```python
import requests

url = "http://localhost:8000/query"
headers = {"Authorization": "Bearer your-api-key"}

payload = {
    "query": "What are the key findings?",
    "top_k": 5,
    "use_multimodal": True
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

print(result["answer"])
print(f"Sources: {result['sources']}")
```

### 3. Run Example Script

```bash
python example_query.py
```

## API Endpoints

### POST /upload
Upload and process documents.

**Request:**
- `file`: multipart/form-data

**Response:**
```json
{
  "doc_id": "doc_abc123",
  "filename": "document.pdf",
  "chunks_processed": {
    "text": 10,
    "table": 2,
    "image": 3
  },
  "message": "Document processed successfully in 5.23s"
}
```

### POST /query
Query the RAG system.

**Request:**
```json
{
  "query": "Your question here",
  "top_k": 5,
  "use_multimodal": true
}
```

**Response:**
```json
{
  "answer": "Markdown formatted answer...",
  "sources": [
    {
      "rank": 1,
      "doc_id": "doc_abc123_text_0",
      "relevance_score": 0.95,
      "type": "text"
    }
  ],
  "processing_time": 2.34
}
```

### DELETE /document/{doc_id}
Delete a document and all associated data.

## Switching LLM/MLLM Backends

The system uses an adapter pattern for easy backend switching.

### Option 1: Use Different OpenAI Models

In `config.py`:
```python
OPENAI_MODEL = "gpt-4-turbo"  # or "gpt-3.5-turbo"
```

### Option 2: Add Local LLM Support

Create a new adapter in `app/models/local_llm_adapter.py`:

```python
from .base_adapter import BaseLLMAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLMAdapter(BaseLLMAdapter):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    async def generate_text(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0])
    
    # Implement other methods...
```

Update `app/server/api.py`:
```python
from app.models.local_llm_adapter import LocalLLMAdapter

llm_adapter = LocalLLMAdapter("mistralai/Mistral-7B-Instruct-v0.1")
```

### Option 3: Add LLaVA for Multimodal

```python
from .base_adapter import BaseLLMAdapter
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class LLaVAAdapter(BaseLLMAdapter):
    def __init__(self):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    
    # Implement methods...
```

## Project Structure

```
multimodal_rag/
├── app/
│   ├── __init__.py
│   ├── models/              # LLM/MLLM adapters
│   │   ├── __init__.py
│   │   ├── base_adapter.py  # Base interface
│   │   ├── llm_adapter.py   # Text-only LLM
│   │   └── mllm_adapter.py  # Multimodal LLM
│   ├── utils/               # Core utilities
│   │   ├── __init__.py
│   │   ├── parser.py        # Document parsing
│   │   ├── summarizer.py    # Semantic summarization
│   │   ├── embedder.py      # Embedding & vector store
│   │   └── retriever.py     # Multi-vector retrieval
│   └── server/              # API server
│       ├── __init__.py
│       ├── api.py           # FastAPI endpoints
│       └── auth.py          # Authentication
├── data/                    # Uploaded documents
├── figures/                 # Extracted images
├── chroma_db/              # ChromaDB storage
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── run_server.py          # Server launcher
├── example_query.py       # Example usage
└── README.md              # This file
```

## How It Works

### 1. Document Upload & Processing

```
PDF/DOCX/HTML → Parser → Text Chunks + Tables + Images
                           ↓
                    Summarizer (MLLM)
                           ↓
                Semantic Summaries (for embedding)
                           ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
         Embeddings                  Raw Content
              ↓                           ↓
         ChromaDB                      Redis
     (Summary vectors)           (Original content)
```

### 2. Query & Retrieval

```
User Query → Embedding → ChromaDB Search
                              ↓
                      Top-K IDs Retrieved
                              ↓
                     Redis Lookup (by ID)
                              ↓
              Raw Content: Text + Tables + Images
                              ↓
                     LLM/MLLM Generation
                              ↓
                      Markdown Answer
```

### 3. Multi-Vector Retrieval

- **Embeddings**: Created from semantic summaries (concise, searchable)
- **Storage**: Raw documents preserved separately (accurate, complete)
- **Retrieval**: Search summaries → return original docs → accurate generation

## Advanced Configuration

### Chunking Strategy

In `config.py`:
```python
CHUNK_SIZE = 1000      # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
```

### Embedding Models

In `config.py`:
```python
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Fast
# Or use: "all-mpnet-base-v2"  # More accurate
# Or use: "BAAI/bge-large-en-v1.5"  # State-of-the-art
```

### Top-K Results

In `config.py`:
```python
TOP_K_RESULTS = 5  # Number of results to retrieve
```

## Testing

### Unit Tests

```bash
pytest tests/
```

### Load Testing

```bash
locust -f tests/load_test.py
```

## Monitoring

The system includes Prometheus metrics at `/metrics`:

- Request count
- Response time
- Error rate
- Document processing time

## Troubleshooting

### Redis Connection Error

```bash
# Start Redis
redis-server

# Check Redis status
redis-cli ping  # Should return PONG
```

### CUDA Out of Memory

For local models, reduce batch size or use CPU:

```python
device = "cpu"  # Instead of "cuda"
```

### Tesseract Not Found

```bash
# Install Tesseract
sudo apt-get install tesseract-ocr

# Set path in config if needed
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

## Performance Optimization

1. **Use GPU**: For local models, enable CUDA
2. **Batch Processing**: Process multiple chunks together
3. **Caching**: Enable Redis caching for embeddings
4. **Async Operations**: All I/O is async for better performance

## Security

- API key authentication
- JWT token support
- Rate limiting (configure in production)
- Input validation
- Secure file handling

## License

MIT License - see LICENSE file

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## Support

For issues and questions:
- GitHub Issues
- Documentation: `http://localhost:8000/docs`

## Acknowledgments

- FastAPI for the web framework
- ChromaDB for vector storage
- OpenAI for LLM/MLLM capabilities
- Sentence Transformers for embeddings
"""

print("=" * 80)
print("MULTI-MODAL RAG SYSTEM - COMPLETE CODEBASE GENERATED")
print("=" * 80)
print("\\nAll files have been generated. To use this system:\\n")
print("1. Create project directory: mkdir multimodal_rag && cd multimodal_rag")
print("2. Save all files in their respective locations")
print("3. Create virtual environment: python -m venv venv")
print("4. Activate: source venv/bin/activate")
print("5. Install: pip install -r requirements.txt")
print("6. Configure .env file with your API keys")
print("7. Start Redis: redis-server")
print("8. Run server: python run_server.py")
print("9. Test: python example_query.py\\n")
print("=" * 80)
        
        prompt = prompts.get(content_type, prompts["text"])
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, semantic summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            return content[:500]  # Fallback to truncation

