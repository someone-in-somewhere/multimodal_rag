"""
Multi-Vector Retriever
Stores and retrieves raw documents from Redis
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import redis

from config import settings

logger = logging.getLogger(__name__)


class MultiVectorRetriever:
    """
    Manages raw document storage and retrieval using Redis
    
    Storage strategy:
    - ChromaDB: Stores embeddings (summaries) for fast search
    - Redis: Stores full raw content for accurate generation
    
    Key format:
    - doc:{doc_id}:{item_id} → Raw content
    - doc_meta:{doc_id} → Document metadata
    """
    
    def __init__(self):
        """Initialize the retriever"""
        self.redis_client = None
        self.is_initialized = False
        
        logger.info("MultiVectorRetriever created (not initialized)")
    
    async def initialize(self):
        """Initialize Redis connection"""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing MultiVectorRetriever...")
            
            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                decode_responses=True,  # Auto-decode responses to strings
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            
            self.is_initialized = True
            logger.info("✅ MultiVectorRetriever initialized (Redis connected)")
        
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize MultiVectorRetriever: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None
        
        self.is_initialized = False
        logger.info("🧹 MultiVectorRetriever cleaned up")
    
    async def store_raw_documents(
        self,
        doc_id: str,
        summaries: List[Dict[str, Any]],
        filename: str
    ):
        """
        Store raw documents in Redis
        
        Args:
            doc_id: Document ID
            summaries: List of summary dictionaries with raw content
            filename: Original filename
        """
        if not self.is_initialized:
            await self.initialize()
        
        logger.info(f"💾 Storing {len(summaries)} items in Redis for doc {doc_id}")
        
        try:
            # Use pipeline for batch operations (more efficient)
            pipe = self.redis_client.pipeline()
            
            # Store each item
            for item in summaries:
                key = f"doc:{doc_id}:{item['id']}"
                
                data = {
                    'id': item['id'],
                    'type': item['type'],
                    'raw': item['raw'],
                    'summary': item['summary']
                }
                
                # Add path for images
                if item['type'] == 'image' and 'path' in item:
                    data['path'] = item['path']
                
                pipe.set(key, json.dumps(data))
            
            # Store metadata
            meta_key = f"doc_meta:{doc_id}"
            meta_data = {
                'doc_id': doc_id,
                'filename': filename,
                'item_count': len(summaries),
                'chunks': {
                    'text': sum(1 for s in summaries if s['type'] == 'text'),
                    'table': sum(1 for s in summaries if s['type'] == 'table'),
                    'image': sum(1 for s in summaries if s['type'] == 'image')
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            pipe.set(meta_key, json.dumps(meta_data))
            
            # Execute all commands
            pipe.execute()
            
            logger.info(f"✅ Stored {len(summaries)} items in Redis")
        
        except Exception as e:
            logger.error(f"❌ Failed to store documents in Redis: {e}")
            raise
    
    async def retrieve_raw_documents(self, ids: List[str]) -> Dict[str, List[str]]:
        """
        Retrieve raw documents from Redis
        
        Args:
            ids: List of document IDs (format: doc_id_item_id)
        
        Returns:
            Dictionary with text_chunks, table_chunks, image_chunks
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not ids:
            return {
                'text_chunks': [],
                'table_chunks': [],
                'image_chunks': []
            }
        
        logger.info(f"🔍 Retrieving {len(ids)} items from Redis")
        
        try:
            text_chunks = []
            table_chunks = []
            image_chunks = []
            
            # Use pipeline for batch get
            pipe = self.redis_client.pipeline()
            
            # Build Redis keys
            keys = []
            for item_id in ids:
                # item_id format: "doc_abc123_text_0"
                # Redis key format: "doc:doc_abc123:text_0"
                # So we need to replace first underscore with colon
                parts = item_id.split('_', 1)
                if len(parts) == 2:
                    redis_key = f"doc:{parts[0]}_{parts[1]}"
                else:
                    redis_key = f"doc:{item_id}"
                
                keys.append(redis_key)
            
            # Get all documents
            for key in keys:
                pipe.get(key)
            
            results = pipe.execute()
            
            # Process results
            for data_str in results:
                if data_str:
                    try:
                        item = json.loads(data_str)
                        
                        if item['type'] == 'text':
                            text_chunks.append(item['raw'])
                        elif item['type'] == 'table':
                            table_chunks.append(item['raw'])
                        elif item['type'] == 'image':
                            image_chunks.append(item['raw'])
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode Redis data: {e}")
                        continue
            
            logger.info(
                f"✅ Retrieved {len(text_chunks)} text, {len(table_chunks)} tables, "
                f"{len(image_chunks)} images"
            )
            
            return {
                'text_chunks': text_chunks,
                'table_chunks': table_chunks,
                'image_chunks': image_chunks
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to retrieve documents from Redis: {e}")
            raise
    
    async def delete_document(self, doc_id: str):
        """
        Delete all items for a document
        
        Args:
            doc_id: Document ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Find all keys for this document
            pattern = f"doc:{doc_id}:*"
            keys = list(self.redis_client.scan_iter(match=pattern))
            
            # Also delete metadata
            meta_key = f"doc_meta:{doc_id}"
            if self.redis_client.exists(meta_key):
                keys.append(meta_key)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"✅ Deleted {len(keys)} keys for doc {doc_id}")
            else:
                logger.warning(f"No keys found for doc {doc_id}")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete document {doc_id}: {e}")
            raise
    
    async def delete_all_documents(self):
        """Delete all documents from Redis"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Find all document keys
            doc_keys = list(self.redis_client.scan_iter(match="doc:*"))
            meta_keys = list(self.redis_client.scan_iter(match="doc_meta:*"))
            
            all_keys = doc_keys + meta_keys
            
            if all_keys:
                self.redis_client.delete(*all_keys)
                logger.info(f"✅ Deleted {len(all_keys)} keys from Redis")
            else:
                logger.info("No documents to delete")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete all documents: {e}")
            raise
    
    async def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents with metadata
        
        Returns:
            List of document metadata dictionaries
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Find all metadata keys
            meta_keys = list(self.redis_client.scan_iter(match="doc_meta:*"))
            
            documents = []
            
            for key in meta_keys:
                try:
                    meta_str = self.redis_client.get(key)
                    if meta_str:
                        meta = json.loads(meta_str)
                        documents.append(meta)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode metadata for {key}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            documents.sort(
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            
            logger.info(f"📋 Found {len(documents)} documents")
            
            return documents
        
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            raise
    
    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document metadata or None if not found
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            meta_key = f"doc_meta:{doc_id}"
            meta_str = self.redis_client.get(meta_key)
            
            if meta_str:
                return json.loads(meta_str)
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Failed to get metadata for {doc_id}: {e}")
            return None
