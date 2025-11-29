"""
app/models/base_adapter.py
Base adapter class for LLM interfaces
Defines the common interface that all LLM adapters must implement
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Enum defining model capabilities"""
    TEXT_GENERATION = "text_generation"
    MULTIMODAL = "multimodal"
    VISION = "vision"
    EMBEDDING = "embedding"


class BaseLLMAdapter(ABC):
    """
    Abstract base class for LLM adapters
    
    All LLM adapters (OpenAI, Ollama, etc.) should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, model_name: str = None, **kwargs):
        """
        Initialize the adapter
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.system_prompt = kwargs.get(
            'system_prompt',
            "You are a helpful AI assistant that provides accurate, "
            "detailed answers based on the given context."
        )
        self.is_initialized = False
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    async def initialize(self):
        """
        Initialize the adapter and any required resources
        Should be called before using the adapter
        """
        pass
    
    @abstractmethod
    async def cleanup(self):
        """
        Cleanup resources when shutting down
        Should be called when the adapter is no longer needed
        """
        pass
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text based on a prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def generate_multimodal(
        self,
        text: str,
        images: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text based on multimodal input (text + images + tables)
        
        Args:
            text: Text context
            images: List of base64-encoded images
            tables: List of table content (markdown format)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def summarize(
        self,
        content: str,
        content_type: str = "text",
        max_length: int = 300,
        **kwargs
    ) -> str:
        """
        Summarize content
        
        Args:
            content: Content to summarize
            content_type: Type of content (text, table, image)
            max_length: Maximum length of summary in characters
            **kwargs: Additional parameters
        
        Returns:
            Summary text
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the adapter is healthy and ready to use
        
        Returns:
            Dictionary with health status
        """
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "model": self.model_name,
            "adapter": self.__class__.__name__
        }
    
    def get_capabilities(self) -> List[ModelCapability]:
        """
        Get the capabilities of this adapter
        
        Returns:
            List of ModelCapability enums
        """
        return [ModelCapability.TEXT_GENERATION]
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """
        Check if the adapter supports a specific capability
        
        Args:
            capability: ModelCapability to check
        
        Returns:
            True if supported, False otherwise
        """
        return capability in self.get_capabilities()
    
    def _validate_input(self, text: str, max_length: int = 10000) -> bool:
        """
        Validate input text
        
        Args:
            text: Input text to validate
            max_length: Maximum allowed length
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If input is invalid
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        if len(text) > max_length:
            raise ValueError(f"Input text too long (max {max_length} characters)")
        
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"

