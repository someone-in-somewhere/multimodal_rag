import openai
import base64
from typing import List, Optional
from .base_adapter import BaseLLMAdapter
from config import settings
import structlog

logger = structlog.get_logger()

class MLLMAdapter(BaseLLMAdapter):
    """Multimodal LLM Adapter - GPT-4o"""
    
    def __init__(self, model_name: str = None, **kwargs):
        super().__init__(model_name or "gpt-4o", **kwargs)
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"Initialized MLLM Adapter with model: {self.model_name}")
    
    async def generate_text(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7, **kwargs) -> str:
        return await self.generate_multimodal(text=prompt, images=None, tables=None, max_tokens=max_tokens, temperature=temperature, **kwargs)
    
    async def generate_multimodal(self, text: str, images: Optional[List[str]] = None, tables: Optional[List[str]] = None, max_tokens: int = 2000, temperature: float = 0.7, **kwargs) -> str:
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        content = []
        context_text = f"Context:\n{text}\n\n"
        if tables:
            context_text += "Tables:\n" + "\n\n".join(tables) + "\n\n"
        content.append({"type": "text", "text": context_text})
        if images:
            for img_base64 in images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
        content.append({"type": "text", "text": "Provide a comprehensive answer in Markdown format."})
        messages.append({"role": "user", "content": content})
        
        try:
            response = await self.client.chat.completions.create(model=self.model_name, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise
    
    async def summarize(self, content: str, content_type: str = "text", image_base64: Optional[str] = None, **kwargs) -> str:
        messages = [{"role": "system", "content": "You are an expert at creating summaries."}]
        if content_type == "image" and image_base64:
            messages.append({"role": "user", "content": [{"type": "text", "text": "Describe this image:"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]})
        else:
            prompts = {"text": f"Summarize:\n\n{content}", "table": f"Summarize this table:\n\n{content}"}
            messages.append({"role": "user", "content": prompts.get(content_type, prompts["text"])})
        
        try:
            response = await self.client.chat.completions.create(model=self.model_name, messages=messages, max_tokens=300, temperature=0.3)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return content[:500]
