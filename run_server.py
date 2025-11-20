#!/usr/bin/env python3
"""
Script to run the Multi-modal RAG server.
"""
import sys
import uvicorn
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import settings


def check_dependencies():
    """Check if required services are available."""
    import redis
    import httpx
    
    warnings = []
    
    # Check Redis
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
            db=settings.REDIS_DB,
            socket_connect_timeout=2
        )
        r.ping()
        print("✓ Redis connection successful")
    except Exception as e:
        warnings.append(f"⚠ Redis not available: {e}")
    
    # Check Ollama if using local LLM
    if settings.USE_LOCAL_LLM:
        try:
            response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            if response.status_code == 200:
                print(f"✓ Ollama available at {settings.OLLAMA_BASE_URL}")
            else:
                warnings.append(f"⚠ Ollama responded with status {response.status_code}")
        except Exception as e:
            warnings.append(f"⚠ Ollama not available: {e}")
    
    # Check OpenAI API key
    if not settings.USE_LOCAL_LLM:
        if not settings.OPENAI_API_KEY:
            warnings.append("⚠ OPENAI_API_KEY not set in environment")
        else:
            print("✓ OpenAI API key configured")
    
    return warnings


def get_server_url():
    """Get the actual server URL for display."""
    host = settings.API_HOST
    port = settings.API_PORT
    
    # If listening on 0.0.0.0, show localhost for convenience
    if host == "0.0.0.0":
        display_host = "localhost"
    else:
        display_host = host
    
    return f"http://{display_host}:{port}"


def main():
    """Main entry point."""
    print("=" * 60)
    print(f"🚀 Starting {settings.API_TITLE} v{settings.API_VERSION}")
    print("=" * 60)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    warnings = check_dependencies()
    
    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  {warning}")
        print("\nServer will start, but some features may not work.\n")
    else:
        print("\n✅ All dependencies available\n")
    
    # Display server info
    server_url = get_server_url()
    print("=" * 60)
    print(f"🌐 Server URL: {server_url}")
    print(f"📚 API Docs:   {server_url}/docs")
    print(f"🔄 ReDoc:      {server_url}/redoc")
    print("=" * 60)
    print(f"\n🔧 Configuration:")
    print(f"  - LLM Mode: {'Local (Ollama)' if settings.USE_LOCAL_LLM else 'OpenAI'}")
    if settings.USE_LOCAL_LLM:
        print(f"  - Model: {settings.OLLAMA_MODEL}")
    print(f"  - Embedding: {settings.SENTENCE_TRANSFORMER_MODEL}")
    print(f"  - Vector DB: ChromaDB at {settings.CHROMA_PERSIST_DIR}")
    print(f"  - Document Store: Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    print("=" * 60)
    print("\n💡 Press CTRL+C to stop the server\n")
    
    try:
        uvicorn.run(
            "app.server.api:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## 📝 Các cải tiến:

1. ✅ **Thêm `check_dependencies()`** - kiểm tra Redis, Ollama, OpenAI key
2. ✅ **Better error handling** - try/except cho server startup
3. ✅ **Improved display** - show URL dễ nhìn hơn (localhost thay vì 0.0.0.0)
4. ✅ **Configuration summary** - hiển thị config đang dùng
5. ✅ **Graceful shutdown** - handle Ctrl+C đúng cách
6. ✅ **Better formatting** - dùng emoji và separators cho dễ đọc
7. ✅ **Path handling** - thêm parent dir vào sys.path để import đúng
8. ✅ **Documentation links** - thêm ReDoc link

## 🎯 Output mẫu khi chạy:
```
============================================================
🚀 Starting Multi-modal RAG System v1.0.0
============================================================

📋 Checking dependencies...
✓ Redis connection successful
✓ Ollama available at http://localhost:11434

✅ All dependencies available

============================================================
🌐 Server URL: http://localhost:8000
📚 API Docs:   http://localhost:8000/docs
🔄 ReDoc:      http://localhost:8000/redoc
============================================================

🔧 Configuration:
  - LLM Mode: Local (Ollama)
  - Model: mistral
  - Embedding: all-MiniLM-L6-v2
  - Vector DB: ChromaDB at /path/to/chroma_db
  - Document Store: Redis at localhost:6379
============================================================

💡 Press CTRL+C to stop the server
