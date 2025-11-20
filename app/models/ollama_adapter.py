import httpx
from typing import List, Optional
from .base_adapter import BaseLLMAdapter
import structlog

logger = structlog.get_logger()

class OllamaAdapter(BaseLLMAdapter):
    """Ollama Local LLM Adapter"""
    
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"Initialized Ollama Adapter with model: {model_name}")
    
    async def generate_text(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7, **kwargs) -> str:
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                }
            )
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            raise
    
    async def generate_multimodal(self, text: str, images: Optional[List[str]] = None, 
                                 tables: Optional[List[str]] = None, max_tokens: int = 2000, 
                                 temperature: float = 0.7, **kwargs) -> str:
        full_prompt = f"Context:\n{text}\n\n"
        if tables:
            full_prompt += "Tables:\n" + "\n\n".join(tables) + "\n\n"
        if images:
            full_prompt += f"Note: {len(images)} image(s) available but cannot be processed by text-only model.\n\n"
        full_prompt += "Question: Based on the context above, provide a comprehensive answer in Markdown format.\n\nAnswer:"
        return await self.generate_text(full_prompt, max_tokens, temperature)
    
    async def summarize(self, content: str, content_type: str = "text", **kwargs) -> str:
        prompts = {
            "text": f"Summarize concisely:\n\n{content}\n\nSummary:",
            "table": f"Summarize this table:\n\n{content}\n\nSummary:",
            "image": f"Describe:\n\n{content}\n\nDescription:"
        }
        prompt = prompts.get(content_type, prompts["text"])
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 300}
                }
            )
            result = response.json()
            return result.get("response", content[:500])
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return content[:500]