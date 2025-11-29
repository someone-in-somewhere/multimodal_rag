"""
multimodal_rag/app/utils/retriever.py


Stores and retrieves raw documents from Redis

Storage strategy:
- ChromaDB: Stores embeddings (summaries) for fast search
- Redis: Stores full raw content for accurate generation
"""

import json
import gzip
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import OrderedDict
import time

try:
    import redis.asyncio as aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    # Fallback to sync redis
    import redis
    AIOREDIS_AVAILABLE = False
    logging.warning("redis.asyncio not available, using sync redis (slower)")

from config import settings

logger = logging.getLogger(__name__)


class DocumentCache:
    """
    LRU cache for frequently accessed documents
    
    Avoids Redis roundtrips for hot documents
    """
    
    def __init__(self, maxsize: int = 100):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Add to cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def invalidate(self, key: str):
        """Remove from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 3)
        }


class MultiVectorRetriever:
    """
    Enhanced multi-vector retriever with production-ready features
    
    Key format:
    - doc:{doc_id}:{item_id} → Compressed raw content
    - doc_meta:{doc_id} → Document metadata
    - doc_index:{doc_id} → Index of all item_ids (for fast deletion)
    
    Features:
    - Async Redis operations (non-blocking)
    - Connection pooling
    - Data compression (gzip)
    - LRU cache for hot documents
    - Retry logic
    - Transaction safety
    - Health monitoring
    """
    
    def __init__(
        self,
        enable_compression: bool = True,
        enable_cache: bool = True,
        cache_size: int = 100,
        max_retries: int = 3,
        connection_pool_size: int = 10,
        batch_size: int = 100,
    ):
        """
        Initialize the retriever
        
        Args:
            enable_compression: Enable gzip compression (default: True)
            enable_cache: Enable document cache (default: True)
            cache_size: Maximum cache size (default: 100 docs)
            max_retries: Maximum retry attempts (default: 3)
            connection_pool_size: Redis connection pool size (default: 10)
            batch_size: Batch size for large operations (default: 100)
        """
        self.enable_compression = enable_compression
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        self.batch_size = batch_size
        
        # Components
        self.redis_client = None
        self.connection_pool = None
        self.is_initialized = False
        
        # Cache
        if enable_cache:
            self.cache = DocumentCache(maxsize=cache_size)
            logger.info(f"✅ Document cache enabled (size: {cache_size})")
        else:
            self.cache = None
        
        # Statistics
        self.stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'total_deleted': 0,
            'compression_ratio': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        logger.info(
            f"MultiVectorRetriever v2 created:\n"
            f"  - Async: {AIOREDIS_AVAILABLE}\n"
            f"  - Compression: {enable_compression}\n"
            f"  - Cache: {enable_cache}\n"
            f"  - Pool size: {connection_pool_size}\n"
            f"  - Batch size: {batch_size}"
        )
    
    async def initialize(self):
        """
        Initialize Redis connection with pooling
        
        IMPROVED: Async Redis, connection pooling
        """
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing MultiVectorRetriever v2...")
            
            if AIOREDIS_AVAILABLE:
                # Use async Redis with connection pooling
                self.connection_pool = aioredis.ConnectionPool.from_url(
                    f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    max_connections=10,
                    decode_responses=False,  # We handle encoding/decoding ourselves
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                
                self.redis_client = aioredis.Redis(connection_pool=self.connection_pool)
                
                # Test connection
                await self.redis_client.ping()
                logger.info("✅ Async Redis connected with connection pool")
            
            else:
                # Fallback to sync Redis
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    decode_responses=False,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                
                self.redis_client.ping()
                logger.info("✅ Sync Redis connected (fallback)")
            
            self.is_initialized = True
            logger.info("✅ MultiVectorRetriever v2 initialized")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize MultiVectorRetriever: {e}", exc_info=True)
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            if AIOREDIS_AVAILABLE:
                await self.redis_client.close()
                if self.connection_pool:
                    await self.connection_pool.disconnect()
            else:
                self.redis_client.close()
            
            self.redis_client = None
            self.connection_pool = None
        
        if self.cache:
            self.cache.clear()
        
        self.is_initialized = False
        logger.info("🧹 MultiVectorRetriever v2 cleaned up")
    
    async def store_raw_documents(
        self,
        doc_id: str,
        summaries: List[Dict[str, Any]],
        filename: str
    ):
        """
        Store raw documents in Redis with compression and transaction safety
        
        IMPROVED: Compression, retry logic, atomic operations
        
        Args:
            doc_id: Document ID
            summaries: List of summary dictionaries with raw content
            filename: Original filename
        """
        if not self.is_initialized:
            await self.initialize()
        
        logger.info(f"💾 Storing {len(summaries)} items in Redis for doc {doc_id}")
        
        start_time = time.time()
        total_uncompressed = 0
        total_compressed = 0
        
        # Store with retry logic
        for attempt in range(self.max_retries):
            try:
                if AIOREDIS_AVAILABLE:
                    await self._store_async(doc_id, summaries, filename)
                else:
                    await asyncio.to_thread(
                        self._store_sync,
                        doc_id,
                        summaries,
                        filename
                    )
                
                # Calculate compression stats
                if self.enable_compression:
                    for item in summaries:
                        raw_data = json.dumps(item)
                        total_uncompressed += len(raw_data)
                        total_compressed += len(self._compress(raw_data))
                    
                    compression_ratio = total_compressed / total_uncompressed if total_uncompressed > 0 else 1.0
                    self.stats['compression_ratio'] = compression_ratio
                
                # Update stats
                self.stats['total_stored'] += len(summaries)
                
                elapsed = time.time() - start_time
                
                if self.enable_compression:
                    logger.info(
                        f"✅ Stored {len(summaries)} items in Redis "
                        f"(compression: {compression_ratio:.1%}, saved {total_uncompressed - total_compressed} bytes) "
                        f"in {elapsed:.2f}s"
                    )
                else:
                    logger.info(f"✅ Stored {len(summaries)} items in Redis in {elapsed:.2f}s")
                
                return  # Success
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ Failed to store after {self.max_retries} attempts: {e}")
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"Store attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def _store_async(
        self,
        doc_id: str,
        summaries: List[Dict[str, Any]],
        filename: str
    ):
        """Async Redis storage (NEW)"""
        # Use pipeline for atomic batch operations
        async with self.redis_client.pipeline(transaction=True) as pipe:
            item_ids = []
            
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
                
                # Serialize and compress
                json_data = json.dumps(data)
                
                if self.enable_compression:
                    stored_data = self._compress(json_data)
                else:
                    stored_data = json_data.encode('utf-8')
                
                pipe.set(key, stored_data)
                item_ids.append(item['id'])
            
            # Store index (for fast deletion)
            index_key = f"doc_index:{doc_id}"
            pipe.set(index_key, json.dumps(item_ids))
            
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
                'timestamp': datetime.utcnow().isoformat(),
                'compressed': self.enable_compression
            }
            pipe.set(meta_key, json.dumps(meta_data).encode('utf-8'))
            
            # Execute all commands atomically
            await pipe.execute()
    
    def _store_sync(
        self,
        doc_id: str,
        summaries: List[Dict[str, Any]],
        filename: str
    ):
        """Sync Redis storage (fallback)"""
        pipe = self.redis_client.pipeline(transaction=True)
        item_ids = []
        
        # Store each item
        for item in summaries:
            key = f"doc:{doc_id}:{item['id']}"
            
            data = {
                'id': item['id'],
                'type': item['type'],
                'raw': item['raw'],
                'summary': item['summary']
            }
            
            if item['type'] == 'image' and 'path' in item:
                data['path'] = item['path']
            
            json_data = json.dumps(data)
            
            if self.enable_compression:
                stored_data = self._compress(json_data)
            else:
                stored_data = json_data.encode('utf-8')
            
            pipe.set(key, stored_data)
            item_ids.append(item['id'])
        
        # Store index
        index_key = f"doc_index:{doc_id}"
        pipe.set(index_key, json.dumps(item_ids))
        
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
            'timestamp': datetime.utcnow().isoformat(),
            'compressed': self.enable_compression
        }
        pipe.set(meta_key, json.dumps(meta_data).encode('utf-8'))
        
        # Execute
        pipe.execute()
    
    async def retrieve_raw_documents(self, ids: List[str]) -> Dict[str, List[str]]:
        """
        Retrieve raw documents from Redis with caching
        
        IMPROVED: Correct key parsing, caching, retry logic
        
        Args:
            ids: List of item IDs (format: doc_id_item_id_hash)
        
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
        
        # Check cache first
        cached_items = {}
        ids_to_fetch = []
        
        if self.cache:
            for item_id in ids:
                cached = self.cache.get(item_id)
                if cached:
                    cached_items[item_id] = cached
                else:
                    ids_to_fetch.append(item_id)
        else:
            ids_to_fetch = ids
        
        # Fetch uncached items with retry
        fetched_items = {}
        
        if ids_to_fetch:
            for attempt in range(self.max_retries):
                try:
                    if AIOREDIS_AVAILABLE:
                        fetched_items = await self._retrieve_async(ids_to_fetch)
                    else:
                        fetched_items = await asyncio.to_thread(
                            self._retrieve_sync,
                            ids_to_fetch
                        )
                    
                    # Add to cache
                    if self.cache:
                        for item_id, item_data in fetched_items.items():
                            self.cache.put(item_id, item_data)
                    
                    break  # Success
                
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"❌ Failed to retrieve after {self.max_retries} attempts: {e}")
                        raise
                    
                    wait_time = 2 ** attempt
                    logger.warning(f"Retrieve attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
        
        # Combine cached and fetched
        all_items = {**cached_items, **fetched_items}
        
        # Organize by type
        text_chunks = []
        table_chunks = []
        image_chunks = []
        
        for item_id in ids:
            item = all_items.get(item_id)
            if item:
                if item['type'] == 'text':
                    text_chunks.append(item['raw'])
                elif item['type'] == 'table':
                    table_chunks.append(item['raw'])
                elif item['type'] == 'image':
                    image_chunks.append(item['raw'])
        
        # Update stats
        self.stats['total_retrieved'] += len(ids)
        if self.cache:
            cache_stats = self.cache.get_stats()
            self.stats['cache_hits'] = cache_stats['hits']
            self.stats['cache_misses'] = cache_stats['misses']
        
        logger.info(
            f"✅ Retrieved {len(text_chunks)} text, {len(table_chunks)} tables, "
            f"{len(image_chunks)} images "
            f"({len(cached_items)} from cache, {len(fetched_items)} from Redis)"
        )
        
        return {
            'text_chunks': text_chunks,
            'table_chunks': table_chunks,
            'image_chunks': image_chunks
        }
    
    async def _retrieve_async(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Async retrieval (NEW)
        
        FIXED: Correct key parsing
        """
        items = {}
        
        # Build Redis keys (FIXED LOGIC)
        redis_keys = []
        for item_id in ids:
            # item_id format: "doc_abc123_chunk_0_a1b2c3"
            # Need to extract: doc_id="doc_abc123" and item_part="chunk_0_a1b2c3"
            # Redis key: "doc:doc_abc123:chunk_0_a1b2c3"
            
            redis_key = self._item_id_to_redis_key(item_id)
            redis_keys.append((item_id, redis_key))
        
        # Batch get with pipeline
        async with self.redis_client.pipeline(transaction=False) as pipe:
            for _, redis_key in redis_keys:
                pipe.get(redis_key)
            
            results = await pipe.execute()
        
        # Process results
        for (item_id, _), data_bytes in zip(redis_keys, results):
            if data_bytes:
                try:
                    # Decompress if needed
                    if self.enable_compression:
                        json_str = self._decompress(data_bytes)
                    else:
                        json_str = data_bytes.decode('utf-8')
                    
                    item = json.loads(json_str)
                    items[item_id] = item
                
                except Exception as e:
                    logger.warning(f"Failed to decode item {item_id}: {e}")
        
        return items
    
    def _retrieve_sync(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Sync retrieval (fallback)"""
        items = {}
        
        # Build Redis keys
        redis_keys = []
        for item_id in ids:
            redis_key = self._item_id_to_redis_key(item_id)
            redis_keys.append((item_id, redis_key))
        
        # Batch get
        pipe = self.redis_client.pipeline(transaction=False)
        for _, redis_key in redis_keys:
            pipe.get(redis_key)
        
        results = pipe.execute()
        
        # Process results
        for (item_id, _), data_bytes in zip(redis_keys, results):
            if data_bytes:
                try:
                    if self.enable_compression:
                        json_str = self._decompress(data_bytes)
                    else:
                        json_str = data_bytes.decode('utf-8')
                    
                    item = json.loads(json_str)
                    items[item_id] = item
                
                except Exception as e:
                    logger.warning(f"Failed to decode item {item_id}: {e}")
        
        return items
    
    def _item_id_to_redis_key(self, item_id: str) -> str:
        """
        Convert item_id to Redis key
        
        FIXED: Correct parsing logic
        
        Args:
            item_id: e.g., "doc_abc123_chunk_0_a1b2c3"
        
        Returns:
            Redis key: e.g., "doc:doc_abc123:chunk_0_a1b2c3"
        """
        # item_id format: "doc_abc123_chunk_0_a1b2c3"
        # We need to extract doc_id which is everything before the third underscore
        
        parts = item_id.split('_')
        
        if len(parts) < 3:
            # Fallback: simple format
            return f"doc:{item_id}"
        
        # doc_id is first 2 parts: "doc_abc123"
        doc_id = '_'.join(parts[:2])
        
        # item_part is rest: "chunk_0_a1b2c3"
        item_part = '_'.join(parts[2:])
        
        return f"doc:{doc_id}:{item_part}"
    
    async def delete_document(self, doc_id: str):
        """
        Delete all items for a document using index
        
        IMPROVED: Use index for fast deletion, invalidate cache
        
        Args:
            doc_id: Document ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        for attempt in range(self.max_retries):
            try:
                if AIOREDIS_AVAILABLE:
                    await self._delete_async(doc_id)
                else:
                    await asyncio.to_thread(self._delete_sync, doc_id)
                
                # Invalidate cache
                if self.cache:
                    # Clear entire cache (simpler than tracking individual items)
                    self.cache.clear()
                
                # Update stats
                self.stats['total_deleted'] += 1
                
                return  # Success
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ Failed to delete document {doc_id}: {e}")
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"Delete attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def _delete_async(self, doc_id: str):
        """Async deletion (NEW)"""
        # Get index
        index_key = f"doc_index:{doc_id}"
        index_data = await self.redis_client.get(index_key)
        
        keys_to_delete = []
        
        if index_data:
            # Use index for fast deletion
            item_ids = json.loads(index_data.decode('utf-8'))
            
            for item_id in item_ids:
                key = f"doc:{doc_id}:{item_id}"
                keys_to_delete.append(key)
            
            # Also delete index and metadata
            keys_to_delete.append(index_key)
        else:
            # Fallback: scan for keys (slower)
            logger.warning(f"No index found for {doc_id}, using scan (slower)")
            
            cursor = 0
            pattern = f"doc:{doc_id}:*"
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                keys_to_delete.extend(keys)
                
                if cursor == 0:
                    break
        
        # Delete metadata
        meta_key = f"doc_meta:{doc_id}"
        keys_to_delete.append(meta_key.encode('utf-8'))
        
        # Batch delete
        if keys_to_delete:
            # Delete in chunks to avoid large operations
            for i in range(0, len(keys_to_delete), self.batch_size):
                batch = keys_to_delete[i:i + self.batch_size]
                await self.redis_client.delete(*batch)
            
            logger.info(f"✅ Deleted {len(keys_to_delete)} keys for doc {doc_id}")
        else:
            logger.warning(f"No keys found for doc {doc_id}")
    
    def _delete_sync(self, doc_id: str):
        """Sync deletion (fallback)"""
        # Get index
        index_key = f"doc_index:{doc_id}"
        index_data = self.redis_client.get(index_key)
        
        keys_to_delete = []
        
        if index_data:
            item_ids = json.loads(index_data.decode('utf-8'))
            
            for item_id in item_ids:
                key = f"doc:{doc_id}:{item_id}"
                keys_to_delete.append(key)
            
            keys_to_delete.append(index_key)
        else:
            logger.warning(f"No index found for {doc_id}, using scan")
            
            pattern = f"doc:{doc_id}:*"
            keys = list(self.redis_client.scan_iter(match=pattern))
            keys_to_delete.extend(keys)
        
        # Delete metadata
        meta_key = f"doc_meta:{doc_id}"
        keys_to_delete.append(meta_key.encode('utf-8'))
        
        # Batch delete
        if keys_to_delete:
            for i in range(0, len(keys_to_delete), self.batch_size):
                batch = keys_to_delete[i:i + self.batch_size]
                self.redis_client.delete(*batch)
            
            logger.info(f"✅ Deleted {len(keys_to_delete)} keys for doc {doc_id}")
        else:
            logger.warning(f"No keys found for doc {doc_id}")
    
    async def delete_all_documents(self):
        """
        Delete all documents from Redis
        
        IMPROVED: Batch deletion for large datasets
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if AIOREDIS_AVAILABLE:
                await self._delete_all_async()
            else:
                await asyncio.to_thread(self._delete_all_sync)
            
            # Clear cache
            if self.cache:
                self.cache.clear()
            
            logger.info("✅ All documents deleted from Redis")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete all documents: {e}")
            raise
    
    async def _delete_all_async(self):
        """Async delete all"""
        all_keys = []
        
        # Scan for all document keys
        for pattern in ["doc:*", "doc_meta:*", "doc_index:*"]:
            cursor = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                all_keys.extend(keys)
                
                if cursor == 0:
                    break
        
        # Batch delete
        if all_keys:
            for i in range(0, len(all_keys), self.batch_size):
                batch = all_keys[i:i + self.batch_size]
                await self.redis_client.delete(*batch)
            
            logger.info(f"Deleted {len(all_keys)} keys from Redis")
    
    def _delete_all_sync(self):
        """Sync delete all"""
        all_keys = []
        
        for pattern in ["doc:*", "doc_meta:*", "doc_index:*"]:
            keys = list(self.redis_client.scan_iter(match=pattern))
            all_keys.extend(keys)
        
        if all_keys:
            for i in range(0, len(all_keys), self.batch_size):
                batch = all_keys[i:i + self.batch_size]
                self.redis_client.delete(*batch)
            
            logger.info(f"Deleted {len(all_keys)} keys from Redis")
    
    async def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents with metadata
        
        Returns:
            List of document metadata dictionaries
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if AIOREDIS_AVAILABLE:
                return await self._list_all_async()
            else:
                return await asyncio.to_thread(self._list_all_sync)
        
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            raise
    
    async def _list_all_async(self) -> List[Dict[str, Any]]:
        """Async list all"""
        documents = []
        
        # Scan for metadata keys
        cursor = 0
        pattern = "doc_meta:*"
        
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            # Get all metadata
            if keys:
                async with self.redis_client.pipeline(transaction=False) as pipe:
                    for key in keys:
                        pipe.get(key)
                    
                    results = await pipe.execute()
                
                for meta_bytes in results:
                    if meta_bytes:
                        try:
                            meta = json.loads(meta_bytes.decode('utf-8'))
                            documents.append(meta)
                        except Exception as e:
                            logger.warning(f"Failed to decode metadata: {e}")
            
            if cursor == 0:
                break
        
        # Sort by timestamp
        documents.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        logger.info(f"📋 Found {len(documents)} documents")
        
        return documents
    
    def _list_all_sync(self) -> List[Dict[str, Any]]:
        """Sync list all"""
        documents = []
        
        meta_keys = list(self.redis_client.scan_iter(match="doc_meta:*"))
        
        for key in meta_keys:
            try:
                meta_bytes = self.redis_client.get(key)
                if meta_bytes:
                    meta = json.loads(meta_bytes.decode('utf-8'))
                    documents.append(meta)
            except Exception as e:
                logger.warning(f"Failed to decode metadata: {e}")
        
        documents.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        logger.info(f"📋 Found {len(documents)} documents")
        
        return documents
    
    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            meta_key = f"doc_meta:{doc_id}"
            
            if AIOREDIS_AVAILABLE:
                meta_bytes = await self.redis_client.get(meta_key)
            else:
                meta_bytes = await asyncio.to_thread(self.redis_client.get, meta_key)
            
            if meta_bytes:
                return json.loads(meta_bytes.decode('utf-8'))
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Failed to get metadata for {doc_id}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics
        
        NEW: Comprehensive stats including compression and cache
        """
        stats = {
            "redis": {
                "connected": self.is_initialized,
                "async": AIOREDIS_AVAILABLE,
            },
            "features": {
                "compression": self.enable_compression,
                "cache": self.enable_cache,
            },
            "operations": {
                "total_stored": self.stats['total_stored'],
                "total_retrieved": self.stats['total_retrieved'],
                "total_deleted": self.stats['total_deleted'],
            }
        }
        
        if self.enable_compression:
            stats["compression"] = {
                "ratio": self.stats['compression_ratio'],
                "savings_percent": (1 - self.stats['compression_ratio']) * 100
            }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for monitoring
        
        NEW: Redis health monitoring
        """
        health = {
            "healthy": False,
            "redis_connected": False,
            "latency_ms": None,
            "error": None
        }
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Measure latency
            start = time.time()
            
            if AIOREDIS_AVAILABLE:
                await self.redis_client.ping()
            else:
                await asyncio.to_thread(self.redis_client.ping)
            
            latency = (time.time() - start) * 1000
            
            health["healthy"] = True
            health["redis_connected"] = True
            health["latency_ms"] = round(latency, 2)
        
        except Exception as e:
            health["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health
    
    # ====================== Compression Utilities ======================
    
    def _compress(self, data: str) -> bytes:
        """Compress string data with gzip"""
        return gzip.compress(data.encode('utf-8'), compresslevel=6)
    
    def _decompress(self, data: bytes) -> str:
        """Decompress gzip data to string"""
        return gzip.decompress(data).decode('utf-8')

