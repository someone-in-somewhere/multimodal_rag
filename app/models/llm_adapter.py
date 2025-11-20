import openai
from typing import List, Optional
from .base_adapter import BaseLLMAdapter
from config import settings
import structlog

logger = structlog.get_logger()

class LLMAdapter(BaseLLMAdapter):
    """
    Text-only LLM Adapter supporting OpenAI, GPT4All, Vicuna, Mistral, etc.
    This implementation uses OpenAI API (can be adapted for local models).
    """
    
    def __init__(self, model_name: str = None, **kwargs):
        super().__init__(model_name or settings.OPENAI_MODEL, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"Initialized LLM Adapter with model: {self.model_name}")
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text response using LLM."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate, well-structured answers based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    async def generate_multimodal(
        self,
        text: str,
        images: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        For text-only LLM, we concatenate all content as text.
        Tables are already in markdown format.
        Images are described in text.
        """
        full_prompt = f"Context:\n{text}\n\n"
        
        if tables:
            full_prompt += "Tables:\n" + "\n\n".join(tables) + "\n\n"
        
        if images:
            full_prompt += f"Note: {len(images)} image(s) are available but cannot be processed by this text-only model.\n\n"
        
        full_prompt += "Question: Based on the above context, please provide a comprehensive answer.\n\n"
        full_prompt += "Answer in Markdown format with proper formatting for tables and lists."
        
        return await self.generate_text(full_prompt, max_tokens, temperature, **kwargs)
    
    async def summarize(
        self,
        content: str,
        content_type: str = "text",
        **kwargs
    ) -> str:
        """Generate semantic summary for embedding."""
        prompts = {
            "text": f"Provide a concise semantic summary of the following text that captures its key concepts and meaning:\n\n{content}",
            "table": f"Summarize the key information and patterns in this table:\n\n{content}",
            "image": f"Describe the key visual elements and information in this image description:\n\n{content}"
        }
        prompt = prompts.get(content_type, prompts["text"])
        return await self.generate_text(prompt, **kwargs)
