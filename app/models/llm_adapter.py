"""
LLM Adapter for text-only language models
Supports: Ollama (local) and OpenAI (cloud)
"""

import httpx
import logging
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI

from app.models.base_adapter import BaseLLMAdapter, ModelCapability
from config import settings

logger = logging.getLogger(__name__)


class LLMAdapter(BaseLLMAdapter):
    """
    Adapter for text-only LLMs
    Automatically switches between Ollama and OpenAI based on settings
    """
    
    def __init__(self, model_name: str = None, **kwargs):
        """
        Initialize LLM adapter
        
        Args:
            model_name: Model name (optional, uses settings if not provided)
            **kwargs: Additional configuration
        """
        # Determine backend
        if settings.USE_LOCAL_LLM:
            self.backend = "ollama"
            self.model_name = model_name or settings.OLLAMA_MODEL
            self.ollama_base_url = settings.OLLAMA_BASE_URL
            self.openai_client = None
        else:
            self.backend = "openai"
            self.model_name = model_name or settings.OPENAI_MODEL
            self.ollama_base_url = None
            self.openai_client = None  # Will be initialized in initialize()
        
        super().__init__(self.model_name, **kwargs)
        
        logger.info(f"LLMAdapter configured for {self.backend} backend")
    
    async def initialize(self):
        """Initialize the adapter and resources"""
        if self.is_initialized:
            return
        
        try:
            if self.backend == "openai":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY not set in environment")
                
                self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ OpenAI client initialized")
            
            elif self.backend == "ollama":
                # Test Ollama connection
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{self.ollama_base_url}/api/tags")
                    
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [m.get("name") for m in models]
                        
                        if self.model_name not in model_names:
                            logger.warning(
                                f"⚠️  Model '{self.model_name}' not found in Ollama. "
                                f"Available: {model_names}"
                            )
                        else:
                            logger.info(f"✅ Ollama model '{self.model_name}' available")
                    else:
                        raise ConnectionError(f"Ollama returned status {response.status_code}")
            
            self.is_initialized = True
            logger.info(f"✅ LLMAdapter initialized successfully ({self.backend})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLMAdapter: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.openai_client:
            await self.openai_client.close()
            self.openai_client = None
        
        self.is_initialized = False
        logger.info("🧹 LLMAdapter cleaned up")
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text based on prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        """
        if not self.is_initialized:
            await self.initialize()
        
        self._validate_input(prompt)
        
        try:
            if self.backend == "ollama":
                return await self._generate_ollama(prompt, max_tokens, temperature, **kwargs)
            else:
                return await self._generate_openai(prompt, max_tokens, temperature, **kwargs)
        
        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            raise
    
    async def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate text using Ollama"""
        try:
            full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "top_p": kwargs.get("top_p", 0.9),
                            "top_k": kwargs.get("top_k", 40)
                        }
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
                result = response.json()
                return result.get("response", "").strip()
        
        except httpx.TimeoutException:
            raise Exception("Ollama request timed out")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    async def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate text using OpenAI"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0)
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
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
        This adapter only supports text. For multimodal, use MLLMAdapter.
        This method combines text and tables as a fallback.
        """
        logger.warning("LLMAdapter doesn't support true multimodal. Using text-only fallback.")
        
        # Combine text and tables
        combined_text = text
        
        if tables:
            combined_text += "\n\nTables:\n" + "\n\n".join(tables)
        
        if images:
            logger.warning("Images provided but LLMAdapter cannot process them. Use MLLMAdapter instead.")
        
        return await self.generate_text(combined_text, max_tokens, temperature, **kwargs)
    
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
            content_type: Type of content (text, table)
            max_length: Maximum length of summary
            **kwargs: Additional parameters
        
        Returns:
            Summary text
        """
        if not self.is_initialized:
            await self.initialize()
        
        self._validate_input(content)
        
        # Create appropriate prompt based on content type
        if content_type == "table":
            prompt = f"""Summarize the key information and trends in this table concisely (max {max_length} characters):

{content}

Summary:"""
        else:  # text
            prompt = f"""Summarize the following text concisely, focusing on key concepts and information (max {max_length} characters):

{content}

Summary:"""
        
        try:
            summary = await self.generate_text(
                prompt,
                max_tokens=int(max_length / 2),  # Approximate tokens
                temperature=0.3,  # Lower temperature for consistency
                **kwargs
            )
            
            # Trim to max_length if needed
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
        
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            raise
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get adapter capabilities"""
        return [ModelCapability.TEXT_GENERATION]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check adapter health"""
        base_health = await super().health_check()
        base_health.update({
            "backend": self.backend,
            "model": self.model_name
        })
        
        # Test connection
        try:
            if self.backend == "ollama":
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.ollama_base_url}/api/tags")
                    base_health["connection"] = "ok" if response.status_code == 200 else "failed"
            elif self.backend == "openai" and self.openai_client:
                # Just check if client exists
                base_health["connection"] = "ok"
            else:
                base_health["connection"] = "not_initialized"
        except Exception as e:
            base_health["connection"] = f"error: {str(e)}"
        
        return base_health
