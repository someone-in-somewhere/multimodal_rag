"""
Configuration module for Multi-modal RAG System V2 (No Auth)
Loads environment variables and provides application settings
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # ========================================================================
    # LLM Configuration
    # ========================================================================
    USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    
    # Ollama (Local LLM)
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma2:4b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # OpenAI (Cloud LLM)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # ========================================================================
    # API Configuration (NO AUTH!)
    # ========================================================================
    # ❌ REMOVED AUTH SETTINGS:
    # - API_KEY
    # - SECRET_KEY
    # - ALGORITHM
    # - ACCESS_TOKEN_EXPIRE_MINUTES
    
    # Server Configuration
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")  # ⚠️ Changed to localhost!
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # ========================================================================
    # Database Configuration
    # ========================================================================
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "multimodal_rag")
    
    # ========================================================================
    # Document Processing Configuration
    # ========================================================================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # ========================================================================
    # Parser V3 Settings (NEW!)
    # ========================================================================
    EXTRACT_PDF_EMBEDDED_IMAGES: bool = os.getenv("EXTRACT_PDF_EMBEDDED_IMAGES", "true").lower() == "true"
    CONVERT_PDF_PAGES_TO_IMAGES: bool = os.getenv("CONVERT_PDF_PAGES_TO_IMAGES", "true").lower() == "true"
    MAX_PDF_PAGES_AS_IMAGES: int = int(os.getenv("MAX_PDF_PAGES_AS_IMAGES", "50"))
    ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "true").lower() == "true"
    
    # ========================================================================
    # Embedder V2 Settings (NEW!)
    # ========================================================================
    EMBEDDER_BATCH_SIZE: int = int(os.getenv("EMBEDDER_BATCH_SIZE", "32"))
    EMBEDDER_CACHE_SIZE: int = int(os.getenv("EMBEDDER_CACHE_SIZE", "1000"))
    EMBEDDER_ENABLE_CACHE: bool = os.getenv("EMBEDDER_ENABLE_CACHE", "true").lower() == "true"
    
    # ========================================================================
    # Retriever V2 Settings (NEW!)
    # ========================================================================
    RETRIEVER_ENABLE_COMPRESSION: bool = os.getenv("RETRIEVER_ENABLE_COMPRESSION", "true").lower() == "true"
    RETRIEVER_ENABLE_CACHE: bool = os.getenv("RETRIEVER_ENABLE_CACHE", "true").lower() == "true"
    RETRIEVER_CACHE_SIZE: int = int(os.getenv("RETRIEVER_CACHE_SIZE", "100"))
    RETRIEVER_CONNECTION_POOL_SIZE: int = int(os.getenv("RETRIEVER_CONNECTION_POOL_SIZE", "10"))
    
    # ========================================================================
    # Summarizer V2 Settings (NEW!)
    # ========================================================================
    SUMMARIZER_ENABLE_CACHE: bool = os.getenv("SUMMARIZER_ENABLE_CACHE", "true").lower() == "true"
    SUMMARIZER_CACHE_SIZE: int = int(os.getenv("SUMMARIZER_CACHE_SIZE", "500"))
    SUMMARIZER_MAX_CONCURRENT: int = int(os.getenv("SUMMARIZER_MAX_CONCURRENT", "5"))
    SUMMARIZER_MAX_RETRIES: int = int(os.getenv("SUMMARIZER_MAX_RETRIES", "3"))
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    SENTENCE_TRANSFORMER_MODEL: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", 
        "all-MiniLM-L6-v2"
    )
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "ViT-B/32")
    
    # ========================================================================
    # Storage Configuration
    # ========================================================================
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    FIGURES_DIR: Path = BASE_DIR / os.getenv("FIGURES_DIR", "figures")
    
    # ========================================================================
    # Optional: Advanced Configuration
    # ========================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_CORS: bool = os.getenv("ENABLE_CORS", "true").lower() == "true"
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "50"))  # MB
    
    # ❌ REMOVED: ENABLE_RATE_LIMIT, RATE_LIMIT_PER_MINUTE
    # Rate limiting is built into V2 components
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


# ============================================================================
# Validation
# ============================================================================

def validate_settings():
    """Validate critical settings"""
    errors = []
    
    # Check if using OpenAI but no API key
    if not settings.USE_LOCAL_LLM and not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when USE_LOCAL_LLM=false")
    
    # ❌ REMOVED: SECRET_KEY length check (no auth)
    
    # Check chunk overlap
    if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    
    # ⚠️ Warning if not localhost
    if settings.API_HOST not in ["127.0.0.1", "localhost"]:
        print("\n" + "⚠️ " * 23)
        print("⚠️  WARNING: API_HOST is set to", settings.API_HOST)
        print("⚠️  NO AUTHENTICATION ENABLED!")
        print("⚠️  Anyone with network access can use the API!")
        print("⚠️  This is ONLY safe for localhost use!")
        print("⚠️ " * 23 + "\n")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Validate on import
validate_settings()
