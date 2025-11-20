"""
Configuration module for Multi-modal RAG System
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
    # API & Security Configuration
    # ========================================================================
    API_KEY: str = os.getenv("API_KEY", "test-api-key")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "my-super-secret-key-min-32-chars-long-abc123")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Server Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
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
    ENABLE_RATE_LIMIT: bool = os.getenv("ENABLE_RATE_LIMIT", "false").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
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
    
    # Check secret key length
    if len(settings.SECRET_KEY) < 32:
        errors.append("SECRET_KEY must be at least 32 characters long")
    
    # Check chunk overlap
    if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Validate on import
validate_settings()
