"""
Server runner for Multi-modal RAG System
Handles startup checks and runs the FastAPI application
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
import redis
import httpx
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_redis():
    """Check if Redis is accessible"""
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        r.ping()
        logger.info("✅ Redis connection: OK")
        return True
    except redis.ConnectionError as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.error("💡 Make sure Redis is running: redis-server")
        return False
    except Exception as e:
        logger.error(f"❌ Redis check error: {e}")
        return False


async def check_ollama():
    """Check if Ollama is accessible (if using local LLM)"""
    if not settings.USE_LOCAL_LLM:
        logger.info("ℹ️  Using OpenAI API (Ollama check skipped)")
        return True
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                logger.info("✅ Ollama connection: OK")
                logger.info(f"📦 Available models: {', '.join(model_names)}")
                
                # Check if configured model exists
                if settings.OLLAMA_MODEL in model_names:
                    logger.info(f"✅ Model '{settings.OLLAMA_MODEL}' is available")
                else:
                    logger.warning(f"⚠️  Model '{settings.OLLAMA_MODEL}' not found")
                    logger.warning(f"💡 Pull it with: ollama pull {settings.OLLAMA_MODEL}")
                    return False
                
                return True
            else:
                logger.error(f"❌ Ollama returned status {response.status_code}")
                return False
                
    except httpx.ConnectError:
        logger.error("❌ Cannot connect to Ollama")
        logger.error("💡 Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        logger.error(f"❌ Ollama check error: {e}")
        return False


def print_startup_info():
    """Print startup information"""
    print("\n" + "="*70)
    print("🚀 Multi-modal RAG System Starting")
    print("="*70)
    print(f"📍 Host: {settings.API_HOST}:{settings.API_PORT}")
    print(f"🔑 API Key: {settings.API_KEY[:10]}...")
    print(f"🤖 LLM Backend: {'Ollama (Local)' if settings.USE_LOCAL_LLM else 'OpenAI (Cloud)'}")
    
    if settings.USE_LOCAL_LLM:
        print(f"   └─ Model: {settings.OLLAMA_MODEL}")
        print(f"   └─ URL: {settings.OLLAMA_BASE_URL}")
    else:
        print(f"   └─ Model: {settings.OPENAI_MODEL}")
    
    print(f"💾 Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    print(f"📊 ChromaDB: {settings.CHROMA_COLLECTION_NAME}")
    print(f"📁 Data Directory: {settings.DATA_DIR}")
    print(f"🖼️  Figures Directory: {settings.FIGURES_DIR}")
    print("="*70)
    print()


async def run_startup_checks():
    """Run all startup checks"""
    logger.info("🔍 Running startup checks...")
    
    checks = [
        ("Redis", check_redis()),
        ("Ollama", check_ollama())
    ]
    
    results = []
    for name, check in checks:
        if asyncio.iscoroutine(check):
            result = await check
        else:
            result = check
        results.append((name, result))
    
    # Check results
    failed = [name for name, result in results if not result]
    
    if failed:
        logger.error(f"\n❌ Startup checks failed: {', '.join(failed)}")
        logger.error("Cannot start server. Please fix the issues above.\n")
        return False
    
    logger.info("✅ All startup checks passed!\n")
    return True


def main():
    """Main entry point"""
    print_startup_info()
    
    # Run startup checks
    checks_passed = asyncio.run(run_startup_checks())
    
    if not checks_passed:
        sys.exit(1)
    
    # Start server
    logger.info("🎉 Starting FastAPI server...\n")
    
    try:
        uvicorn.run(
            "app.server.api:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=True,  # Auto-reload on code changes (development)
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"\n💥 Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
