from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLMAdapter(ABC):
    """
    Base adapter class for LLM/MLLM backends.
    Implement this interface to add new model backends.
    """
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text response from text-only prompt."""
        pass
    
    @abstractmethod
    async def generate_multimodal(
        self,
        text: str,
        images: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response from multimodal inputs (text + images + tables)."""
        pass
    
    @abstractmethod
    async def summarize(
        self,
        content: str,
        content_type: str = "text",
        **kwargs
    ) -> str:
        """Generate semantic summary of content."""
        pass
