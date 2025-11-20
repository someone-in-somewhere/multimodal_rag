import os
from pathlib import Path
from pydantic_settings import BaseSettings

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

    # Local LLM Configuration
    USE_LOCAL_LLM: bool = True
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    FIGURES_DIR: Path = BASE_DIR / "figures"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    USE_LOCAL_LLM: bool = True
    OLLAMA_MODEL: str = "gemma3"
    
    # Chroma Configuration
    CHROMA_PERSIST_DIR: str = "./chroma_db"
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

settings = Settings()

# Create directories
settings.DATA_DIR.mkdir(exist_ok=True)
settings.FIGURES_DIR.mkdir(exist_ok=True)
