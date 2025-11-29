"""
app/models/mllm_adapter.py
Multimodal LLM Adapter for vision-language models
Supports: Ollama (LLaVA) and OpenAI (GPT-4o)
"""

import httpx
import logging
import base64
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI

from app.models.base_adapter import BaseLLMAdapter, ModelCapability
from config import settings

logger = logging.getLogger(__name__)


class MLLMAdapter(BaseLLMAdapter):
    """
    Adapter for Multimodal LLMs (text + images)
    Automatically switches between Ollama (LLaVA) and OpenAI (GPT-4o)
    """
    
    def __init__(self, model_name: str = None, **kwargs):
        """
        Initialize MLLM adapter
        
        Args:
            model_name: Model name (optional)
            **kwargs: Additional configuration
        """
        # Determine backend
        if settings.USE_LOCAL_LLM:
            self.backend = "ollama"
            self.model_name = model_name or "llava"  # Default to LLaVA for multimodal
            self.ollama_base_url = settings.OLLAMA_BASE_URL
            self.openai_client = None
        else:
            self.backend = "openai"
            self.model_name = model_name or "gpt-4o"  # GPT-4o supports vision
            self.ollama_base_url = None
            self.openai_client = None
        
        super().__init__(self.model_name, **kwargs)
        
        self._supports_vision = True
        logger.info(f"MLLMAdapter configured for {self.backend} backend")
    
    async def initialize(self):
        """Initialize the adapter"""
        if self.is_initialized:
            return
        
        try:
            if self.backend == "openai":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY not set")
                
                self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ OpenAI client initialized for multimodal")
            
            elif self.backend == "ollama":
                # Check if LLaVA or similar multimodal model is available
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{self.ollama_base_url}/api/tags")
                    
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [m.get("name") for m in models]
                        
                        # Check for multimodal models
                        multimodal_models = [m for m in model_names if "llava" in m.lower() or "vision" in m.lower()]
                        
                        if not multimodal_models:
                            logger.warning(
                                "⚠️  No multimodal models found in Ollama. "
                                "Consider pulling llava: ollama pull llava"
                            )
                            self._supports_vision = False
                        else:
                            logger.info(f"✅ Multimodal models available: {multimodal_models}")
                            
                            if self.model_name not in model_names:
                                # Use first available multimodal model
                                self.model_name = multimodal_models[0]
                                logger.info(f"Using model: {self.model_name}")
            
            self.is_initialized = True
            logger.info(f"✅ MLLMAdapter initialized ({self.backend})")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize MLLMAdapter: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.openai_client:
            await self.openai_client.close()
            self.openai_client = None
        
        self.is_initialized = False
        logger.info("🧹 MLLMAdapter cleaned up")
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text (same as LLMAdapter for text-only)"""
        if not self.is_initialized:
            await self.initialize()
        
        self._validate_input(prompt)
        
        try:
            if self.backend == "ollama":
                return await self._generate_ollama_text(prompt, max_tokens, temperature, **kwargs)
            else:
                return await self._generate_openai_text(prompt, max_tokens, temperature, **kwargs)
        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
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
        Generate text based on multimodal input
        
        Args:
            text: Text context
            images: List of base64-encoded images
            tables: List of tables in markdown format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self._supports_vision and images:
            logger.warning("Vision not supported, falling back to text-only")
            images = None
        
        # Build combined context
        context = text
        
        if tables:
            context += "\n\nTables:\n" + "\n\n".join(tables)
        
        try:
            if images and self._supports_vision:
                if self.backend == "ollama":
                    return await self._generate_ollama_multimodal(
                        context, images, max_tokens, temperature, **kwargs
                    )
                else:
                    return await self._generate_openai_multimodal(
                        context, images, max_tokens, temperature, **kwargs
                    )
            else:
                # Text only
                return await self.generate_text(context, max_tokens, temperature, **kwargs)
        
        except Exception as e:
            logger.error(f"❌ Multimodal generation failed: {e}")
            raise
    
    async def _generate_ollama_text(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate text using Ollama"""
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
                        "num_predict": max_tokens
                    }
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.status_code}")
            
            result = response.json()
            return result.get("response", "").strip()
    
    async def _generate_ollama_multimodal(
        self,
        text: str,
        images: List[str],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate using Ollama multimodal (LLaVA)"""
        prompt = f"{self.system_prompt}\n\n{text}"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": images,  # Base64 images
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama multimodal error: {response.status_code}")
            
            result = response.json()
            return result.get("response", "").strip()
    
    async def _generate_openai_text(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate text using OpenAI"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_openai_multimodal(
        self,
        text: str,
        images: List[str],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate using OpenAI multimodal (GPT-4o)"""
        # Build content with text and images
        content = [{"type": "text", "text": text}]
        
        # Add images
        for img_base64 in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content}
        ]
        
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()
    
    async def summarize(
        self,
        content: str,
        content_type: str = "text",
        max_length: int = 300,
        image_base64: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Summarize content (including images if provided)
        
        Args:
            content: Text content to summarize
            content_type: Type of content (text, table, image)
            max_length: Maximum summary length
            image_base64: Base64-encoded image (for image type)
            **kwargs: Additional parameters
        
        Returns:
            Summary text
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Build prompt based on content type
        if content_type == "image" and image_base64:
            prompt = f"Describe the key information and visual elements in this image concisely (max {max_length} characters):"
            
            if self._supports_vision:
                return await self.generate_multimodal(
                    text=prompt,
                    images=[image_base64],
                    max_tokens=int(max_length / 2),
                    temperature=0.3
                )
            else:
                return "Image summarization not supported by current model"
        
        elif content_type == "table":
            prompt = f"""Summarize the key information in this table (max {max_length} characters):

{content}

Summary:"""
        else:  # text
            prompt = f"""Summarize concisely (max {max_length} characters):

{content}

Summary:"""
        
        summary = await self.generate_text(
            prompt,
            max_tokens=int(max_length / 2),
            temperature=0.3
        )
        
        # Trim if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get adapter capabilities"""
        caps = [ModelCapability.TEXT_GENERATION, ModelCapability.MULTIMODAL]
        
        if self._supports_vision:
            caps.append(ModelCapability.VISION)
        
        return caps
    
    async def health_check(self) -> Dict[str, Any]:
        """Check adapter health"""
        base_health = await super().health_check()
        base_health.update({
            "backend": self.backend,
            "model": self.model_name,
            "supports_vision": self._supports_vision
        })
        
        return base_health
