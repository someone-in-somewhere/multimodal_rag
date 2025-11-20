"""
Content Summarizer
Creates semantic summaries of text, tables, and images using LLMs
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from app.models.mllm_adapter import MLLMAdapter

logger = logging.getLogger(__name__)


class ContentSummarizer:
    """
    Summarizes content for embedding
    
    Why summarize?
    - Raw content is too long/detailed for embeddings
    - Summaries capture semantic meaning better
    - Improves retrieval quality
    
    Uses MLLM for:
    - Text summarization
    - Table summarization
    - Image description
    """
    
    def __init__(self, mllm_adapter: MLLMAdapter):
        """
        Initialize the summarizer
        
        Args:
            mllm_adapter: Multimodal LLM adapter
        """
        self.mllm = mllm_adapter
        
        logger.info("ContentSummarizer initialized")
    
    async def summarize(
        self,
        content: str,
        content_type: str = "text",
        max_length: int = 300
    ) -> str:
        """
        Summarize text or table content
        
        Args:
            content: Content to summarize
            content_type: Type of content ("text" or "table")
            max_length: Maximum summary length in characters
        
        Returns:
            Summary text
        """
        if not content or not content.strip():
            return ""
        
        try:
            # Truncate very long content to avoid token limits
            if len(content) > 10000:
                logger.warning(f"Content too long ({len(content)} chars), truncating")
                content = content[:10000] + "..."
            
            summary = await self.mllm.summarize(
                content=content,
                content_type=content_type,
                max_length=max_length
            )
            
            return summary.strip()
        
        except Exception as e:
            logger.error(f"❌ Summarization failed for {content_type}: {e}")
            # Fallback: return truncated content
            return content[:max_length] + "..." if len(content) > max_length else content
    
    async def summarize_image(self, image_base64: str, max_length: int = 300) -> str:
        """
        Describe an image
        
        Args:
            image_base64: Base64-encoded image
            max_length: Maximum description length
        
        Returns:
            Image description
        """
        if not image_base64:
            return ""
        
        try:
            description = await self.mllm.summarize(
                content="",
                content_type="image",
                max_length=max_length,
                image_base64=image_base64
            )
            
            return description.strip()
        
        except Exception as e:
            logger.error(f"❌ Image summarization failed: {e}")
            return "Image content (description unavailable)"
    
    async def summarize_text_chunks(
        self,
        chunks: List[str],
        batch_size: int = 5,
        max_length: int = 300
    ) -> List[str]:
        """
        Summarize multiple text chunks concurrently
        
        Args:
            chunks: List of text chunks
            batch_size: Number of chunks to process concurrently
            max_length: Maximum summary length per chunk
        
        Returns:
            List of summaries
        """
        if not chunks:
            return []
        
        logger.info(f"Summarizing {len(chunks)} text chunks (batch_size={batch_size})")
        
        summaries = []
        
        # Process in batches to avoid overwhelming the LLM
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [
                self._summarize_text_chunk(chunk, idx, max_length)
                for idx, chunk in enumerate(batch, start=i)
            ]
            
            # Run concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for idx, result in enumerate(batch_results, start=i):
                if isinstance(result, Exception):
                    logger.error(f"Batch summarization error for chunk {idx}: {result}")
                    # Fallback to truncated original
                    chunk = chunks[idx]
                    summaries.append(
                        chunk[:max_length] + "..." if len(chunk) > max_length else chunk
                    )
                else:
                    summaries.append(result)
        
        logger.info(f"✅ Summarized {len(summaries)} chunks")
        
        return summaries
    
    async def _summarize_text_chunk(
        self,
        chunk: str,
        idx: int,
        max_length: int
    ) -> str:
        """
        Internal helper to summarize a single text chunk
        
        Args:
            chunk: Text chunk
            idx: Chunk index (for logging)
            max_length: Maximum summary length
        
        Returns:
            Summary
        """
        try:
            logger.debug(f"Summarizing chunk {idx}...")
            
            summary = await self.summarize(
                content=chunk,
                content_type="text",
                max_length=max_length
            )
            
            return summary
        
        except Exception as e:
            logger.error(f"Failed to summarize chunk {idx}: {e}")
            raise
    
    async def summarize_tables(
        self,
        tables: List[Dict[str, str]],
        max_length: int = 300
    ) -> List[str]:
        """
        Summarize multiple tables
        
        Args:
            tables: List of table dictionaries with 'content' key
            max_length: Maximum summary length
        
        Returns:
            List of summaries
        """
        if not tables:
            return []
        
        logger.info(f"Summarizing {len(tables)} tables")
        
        summaries = []
        
        for idx, table in enumerate(tables):
            try:
                summary = await self.summarize(
                    content=table['content'],
                    content_type="table",
                    max_length=max_length
                )
                summaries.append(summary)
            
            except Exception as e:
                logger.error(f"Failed to summarize table {idx}: {e}")
                # Fallback
                content = table['content']
                summaries.append(
                    content[:max_length] + "..." if len(content) > max_length else content
                )
        
        logger.info(f"✅ Summarized {len(summaries)} tables")
        
        return summaries
    
    async def summarize_images(
        self,
        images: List[Dict[str, str]],
        max_length: int = 300
    ) -> List[str]:
        """
        Describe multiple images
        
        Args:
            images: List of image dictionaries with 'base64' key
            max_length: Maximum description length
        
        Returns:
            List of descriptions
        """
        if not images:
            return []
        
        logger.info(f"Describing {len(images)} images")
        
        descriptions = []
        
        for idx, image in enumerate(images):
            try:
                description = await self.summarize_image(
                    image_base64=image['base64'],
                    max_length=max_length
                )
                descriptions.append(description)
            
            except Exception as e:
                logger.error(f"Failed to describe image {idx}: {e}")
                descriptions.append("Image content (description unavailable)")
        
        logger.info(f"✅ Described {len(descriptions)} images")
        
        return descriptions
    
    async def summarize_all(
        self,
        text_chunks: List[str],
        tables: List[Dict[str, str]],
        images: List[Dict[str, str]],
        batch_size: int = 5,
        max_length: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Summarize all content types together
        
        Args:
            text_chunks: List of text chunks
            tables: List of table dictionaries
            images: List of image dictionaries
            batch_size: Batch size for text processing
            max_length: Maximum summary length
        
        Returns:
            List of summary dictionaries with 'id', 'summary', 'raw', 'type'
        """
        logger.info(
            f"Summarizing all content: {len(text_chunks)} texts, "
            f"{len(tables)} tables, {len(images)} images"
        )
        
        all_summaries = []
        
        # Summarize text chunks
        if text_chunks:
            text_summaries = await self.summarize_text_chunks(
                text_chunks,
                batch_size=batch_size,
                max_length=max_length
            )
            
            for idx, (chunk, summary) in enumerate(zip(text_chunks, text_summaries)):
                all_summaries.append({
                    'id': f"text_{idx}",
                    'summary': summary,
                    'raw': chunk,
                    'type': 'text'
                })
        
        # Summarize tables
        if tables:
            table_summaries = await self.summarize_tables(tables, max_length)
            
            for table, summary in zip(tables, table_summaries):
                all_summaries.append({
                    'id': table.get('id', 'table_0'),
                    'summary': summary,
                    'raw': table['content'],
                    'type': 'table'
                })
        
        # Describe images
        if images:
            image_descriptions = await self.summarize_images(images, max_length)
            
            for image, description in zip(images, image_descriptions):
                all_summaries.append({
                    'id': image.get('id', 'image_0'),
                    'summary': description,
                    'raw': image['base64'],
                    'path': image.get('path', ''),
                    'type': 'image'
                })
        
        logger.info(f"✅ Total summaries created: {len(all_summaries)}")
        
        return all_summaries
