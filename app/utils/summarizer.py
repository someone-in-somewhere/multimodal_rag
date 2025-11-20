from typing import List, Dict, Any
import structlog
from app.models.mllm_adapter import MLLMAdapter

logger = structlog.get_logger()

class ContentSummarizer:
    """
    Generates semantic summaries for text, tables, and images.
    Summaries are used for embedding while raw content is preserved.
    """
    
    def __init__(self, mllm_adapter: MLLMAdapter):
        self.mllm = mllm_adapter
    
    async def summarize_text_chunks(self, chunks: List[str]) -> List[Dict[str, str]]:
        """Summarize text chunks for embedding."""
        summaries = []
        
        for idx, chunk in enumerate(chunks):
            try:
                summary = await self.mllm.summarize(chunk, content_type="text")
                summaries.append({
                    'id': f"text_{idx}",
                    'summary': summary,
                    'raw': chunk,
                    'type': 'text'
                })
            except Exception as e:
                logger.error(f"Error summarizing text chunk {idx}: {str(e)}")
                summaries.append({
                    'id': f"text_{idx}",
                    'summary': chunk[:500],  # Fallback
                    'raw': chunk,
                    'type': 'text'
                })
        
        logger.info(f"Summarized {len(summaries)} text chunks")
        return summaries
    
    async def summarize_tables(self, tables: List[Dict]) -> List[Dict[str, str]]:
        """Summarize tables for embedding."""
        summaries = []
        
        for table in tables:
            try:
                summary = await self.mllm.summarize(
                    table['content'],
                    content_type="table"
                )
                summaries.append({
                    'id': table['id'],
                    'summary': summary,
                    'raw': table['content'],  # Markdown format
                    'type': 'table'
                })
            except Exception as e:
                logger.error(f"Error summarizing table {table['id']}: {str(e)}")
                summaries.append({
                    'id': table['id'],
                    'summary': table['content'][:500],
                    'raw': table['content'],
                    'type': 'table'
                })
        
        logger.info(f"Summarized {len(summaries)} tables")
        return summaries
    
    async def summarize_images(self, images: List[Dict]) -> List[Dict[str, str]]:
        """Summarize images for embedding using vision capabilities."""
        summaries = []
        
        for image in images:
            try:
                summary = await self.mllm.summarize(
                    content="",
                    content_type="image",
                    image_base64=image['base64']
                )
                summaries.append({
                    'id': image['id'],
                    'summary': summary,
                    'raw': image['base64'],  # Store base64
                    'path': image['path'],
                    'type': 'image'
                })
            except Exception as e:
                logger.error(f"Error summarizing image {image['id']}: {str(e)}")
                summaries.append({
                    'id': image['id'],
                    'summary': f"Image: {image['id']}",
                    'raw': image['base64'],
                    'path': image['path'],
                    'type': 'image'
                })
        
        logger.info(f"Summarized {len(summaries)} images")
        return summaries
