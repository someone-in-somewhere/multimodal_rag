import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    API_TITLE: str = "Multi-modal RAG System"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    API_KEY: str = "test-api-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Local LLM Configuration
    USE_LOCAL_LLM: bool = True
    OLLAMA_MODEL: str = "mistral"  # Changed from gemma3 to mistral for consistency
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.resolve()
    DATA_DIR: Optional[Path] = None
    FIGURES_DIR: Optional[Path] = None
    CHROMA_PERSIST_DIR: Optional[Path] = None
    
    # Chroma Configuration
    CHROMA_COLLECTION_NAME: str = "multimodal_rag"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    
    # Model Configuration
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    CLIP_MODEL: str = "ViT-B/32"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize paths after BASE_DIR is set
        if self.DATA_DIR is None:
            self.DATA_DIR = self.BASE_DIR / "data"
        if self.FIGURES_DIR is None:
            self.FIGURES_DIR = self.BASE_DIR / "figures"
        if self.CHROMA_PERSIST_DIR is None:
            self.CHROMA_PERSIST_DIR = self.BASE_DIR / "chroma_db"
        
        # Create directories
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# Initialize settings
settings = Settings()
