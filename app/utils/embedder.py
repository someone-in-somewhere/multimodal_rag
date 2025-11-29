"""
app/utils/embedder.py

Handles text embeddings and vector storage in ChromaDB
"""

import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from datetime import datetime
import time

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU cache for embeddings
    
    Avoids re-computing embeddings for common queries
    """
    
    def __init__(self, maxsize: int = 1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: List[float]):
        """Add item to cache"""
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.maxsize:
                # Remove oldest item
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
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


class EmbeddingManager:
    """
    Enhanced embedding manager with production-ready features
    
    Features:
    - Batch processing for efficiency
    - Embedding cache for common queries
    - Device management (CPU/GPU)
    - Retry logic for reliability
    - Memory-efficient streaming
    - Progress tracking
    - Hybrid search support
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        enable_cache: bool = True,
        cache_size: int = 1000,
        device: Optional[str] = None,
        max_retries: int = 3,
        enable_progress_logging: bool = True,
    ):
        """
        Initialize the embedding manager
        
        Args:
            batch_size: Batch size for encoding (default: 32)
            enable_cache: Enable embedding cache (default: True)
            cache_size: Maximum cache size (default: 1000)
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_retries: Maximum retry attempts (default: 3)
            enable_progress_logging: Log progress for long operations
        """
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        self.enable_progress_logging = enable_progress_logging
        
        # Components
        self.client = None
        self.collection = None
        self.text_model = None
        self.device = device
        self.is_initialized = False
        
        # Cache
        if enable_cache:
            self.cache = LRUCache(maxsize=cache_size)
            logger.info(f"✅ Embedding cache enabled (size: {cache_size})")
        else:
            self.cache = None
        
        # Statistics
        self.stats = {
            'total_embeddings_created': 0,
            'total_items_stored': 0,
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        logger.info(
            f"EmbeddingManager v2 created:\n"
            f"  - Batch size: {batch_size}\n"
            f"  - Cache: {enable_cache}\n"
            f"  - Max retries: {max_retries}"
        )
    
    async def initialize(self):
        """
        Initialize ChromaDB and embedding model
        
        IMPROVED: Better device management and error handling
        """
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing EmbeddingManager v2...")
            
            # Initialize ChromaDB client
            chroma_settings = ChromaSettings(
                persist_directory=settings.CHROMA_PERSIST_DIR,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.Client(chroma_settings)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=settings.CHROMA_COLLECTION_NAME
                )
                logger.info(f"✅ Using existing collection: {settings.CHROMA_COLLECTION_NAME}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=settings.CHROMA_COLLECTION_NAME,
                    metadata={"description": "Multi-modal RAG embeddings"}
                )
                logger.info(f"✅ Created new collection: {settings.CHROMA_COLLECTION_NAME}")
            
            # Initialize embedding model with device management
            await self._initialize_model()
            
            self.is_initialized = True
            logger.info("✅ EmbeddingManager v2 initialized successfully")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize EmbeddingManager: {e}", exc_info=True)
            raise
    
    async def _initialize_model(self):
        """
        Initialize embedding model with proper device management
        
        NEW: Auto-detect GPU, handle OOM
        """
        logger.info(f"Loading embedding model: {settings.SENTENCE_TRANSFORMER_MODEL}")
        
        # Determine device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                logger.info("ℹ️ CUDA not available, using CPU")
        
        # Load model in thread pool (blocking operation)
        try:
            self.text_model = await asyncio.to_thread(
                self._load_model_sync,
                settings.SENTENCE_TRANSFORMER_MODEL,
                self.device
            )
            
            # Get model info
            embedding_dim = self.text_model.get_sentence_embedding_dimension()
            max_seq_length = self.text_model.max_seq_length
            
            logger.info(
                f"✅ Model loaded:\n"
                f"  - Device: {self.device}\n"
                f"  - Embedding dim: {embedding_dim}\n"
                f"  - Max sequence length: {max_seq_length}"
            )
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device == 'cuda':
                logger.warning("⚠️ GPU OOM, falling back to CPU")
                torch.cuda.empty_cache()
                self.device = 'cpu'
                self.text_model = await asyncio.to_thread(
                    self._load_model_sync,
                    settings.SENTENCE_TRANSFORMER_MODEL,
                    'cpu'
                )
                logger.info("✅ Model loaded on CPU")
            else:
                raise
    
    def _load_model_sync(self, model_name: str, device: str) -> SentenceTransformer:
        """Load model synchronously (runs in thread pool)"""
        model = SentenceTransformer(model_name, device=device)
        return model
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.text_model and self.device == 'cuda':
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        self.client = None
        self.collection = None
        self.text_model = None
        self.is_initialized = False
        
        if self.cache:
            self.cache.clear()
        
        logger.info("🧹 EmbeddingManager v2 cleaned up")
    
    async def embed_texts_batch(
        self,
        texts: List[str],
        show_progress: bool = None
    ) -> List[List[float]]:
        """
        Create embeddings for multiple texts with batch processing
        
        NEW: Proper batching, caching, non-blocking execution
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress (None = auto based on size)
        
        Returns:
            List of embeddings
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not texts:
            return []
        
        # Auto-enable progress for large batches
        if show_progress is None:
            show_progress = len(texts) > 100 and self.enable_progress_logging
        
        if show_progress:
            logger.info(f"Creating embeddings for {len(texts)} texts...")
        
        # Check cache first
        embeddings = []
        texts_to_embed = []
        cache_indices = []
        
        if self.cache:
            for idx, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self.cache.get(cache_key)
                
                if cached_embedding:
                    embeddings.append((idx, cached_embedding))
                else:
                    texts_to_embed.append(text)
                    cache_indices.append(idx)
        else:
            texts_to_embed = texts
            cache_indices = list(range(len(texts)))
        
        # Encode texts that are not cached
        if texts_to_embed:
            new_embeddings = await self._encode_batch(
                texts_to_embed,
                show_progress=show_progress
            )
            
            # Add to cache and result
            for idx, text, embedding in zip(cache_indices, texts_to_embed, new_embeddings):
                if self.cache:
                    cache_key = self._get_cache_key(text)
                    self.cache.put(cache_key, embedding)
                
                embeddings.append((idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = [emb for _, emb in embeddings]
        
        # Update stats
        self.stats['total_embeddings_created'] += len(texts_to_embed)
        if self.cache:
            cache_stats = self.cache.get_stats()
            self.stats['cache_hits'] = cache_stats['hits']
            self.stats['cache_misses'] = cache_stats['misses']
        
        if show_progress:
            logger.info(
                f"✅ Created {len(texts_to_embed)} new embeddings, "
                f"{len(texts) - len(texts_to_embed)} from cache"
            )
        
        return result
    
    async def _encode_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Encode texts in batches (runs in thread pool)
        
        NEW: Non-blocking batch encoding with progress tracking
        """
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        all_embeddings = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Run encoding in thread pool (blocking operation)
            batch_embeddings = await asyncio.to_thread(
                self._encode_sync,
                batch_texts
            )
            
            all_embeddings.extend(batch_embeddings)
            
            # Progress logging
            if show_progress and total_batches > 1:
                progress = (batch_idx + 1) / total_batches * 100
                logger.debug(
                    f"Encoding progress: {batch_idx + 1}/{total_batches} batches "
                    f"({progress:.1f}%)"
                )
        
        return all_embeddings
    
    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous batch encoding (runs in thread pool)
        
        Args:
            texts: Batch of texts
        
        Returns:
            List of embeddings
        """
        try:
            # Encode batch
            embeddings = self.text_model.encode(
                texts,
                batch_size=len(texts),  # Encode all at once
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )
            
            return embeddings.tolist()
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device == 'cuda':
                logger.warning("⚠️ GPU OOM during encoding, falling back to CPU")
                torch.cuda.empty_cache()
                
                # Retry on CPU
                self.text_model.to('cpu')
                self.device = 'cpu'
                
                embeddings = self.text_model.encode(
                    texts,
                    batch_size=len(texts),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                
                return embeddings.tolist()
            else:
                raise
    
    async def embed_and_store(
        self,
        summaries: List[Dict[str, Any]],
        doc_id: str
    ) -> Dict[str, int]:
        """
        Create embeddings and store in ChromaDB with retry logic
        
        IMPROVED: Batch processing, retry logic, better error handling
        
        Args:
            summaries: List of summary dictionaries with 'id', 'summary', 'type'
            doc_id: Document ID
        
        Returns:
            Dictionary with counts by type
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not summaries:
            logger.warning("No summaries provided for embedding")
            return {"text": 0, "table": 0, "image": 0}
        
        start_time = time.time()
        logger.info(f"Processing {len(summaries)} items for doc {doc_id}...")
        
        # Prepare data
        texts = [item['summary'] for item in summaries]
        
        # Batch embed all texts (with caching!)
        embeddings = await self.embed_texts_batch(texts, show_progress=True)
        
        # Prepare ChromaDB data
        documents = []
        metadatas = []
        ids = []
        counts = {"text": 0, "table": 0, "image": 0}
        
        for item, embedding in zip(summaries, embeddings):
            documents.append(item['summary'])
            metadatas.append({
                'doc_id': doc_id,
                'item_id': item['id'],
                'type': item['type']
            })
            ids.append(f"{doc_id}_{item['id']}")
            
            # Count by type
            item_type = item['type']
            if item_type in counts:
                counts[item_type] += 1
        
        # Store in ChromaDB with retry logic
        await self._store_with_retry(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Update stats
        self.stats['total_items_stored'] += len(summaries)
        
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"✅ Stored {len(embeddings)} embeddings for doc {doc_id} "
            f"(text: {counts['text']}, table: {counts['table']}, image: {counts['image']}) "
            f"in {elapsed_time:.2f}s"
        )
        
        return counts
    
    async def _store_with_retry(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """
        Store embeddings with exponential backoff retry
        
        NEW: Retry logic for reliability
        """
        for attempt in range(self.max_retries):
            try:
                # Run in thread pool (blocking I/O)
                await asyncio.to_thread(
                    self.collection.add,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                return  # Success
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ Failed to store after {self.max_retries} attempts: {e}")
                    raise
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"⚠️ Store attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
    
    async def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents
        
        IMPROVED: Cached query embeddings, retry logic
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_dict: Optional metadata filter
        
        Returns:
            Dictionary with ids, distances, metadatas, documents
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        try:
            # Create query embedding (with cache!)
            query_embeddings = await self.embed_texts_batch([query_text])
            query_embedding = query_embeddings[0]
            
            # Search in ChromaDB (with retry)
            results = await self._query_with_retry(
                query_embedding=query_embedding,
                n_results=n_results,
                filter_dict=filter_dict
            )
            
            # Update stats
            self.stats['total_queries'] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Query failed: {e}", exc_info=True)
            raise
    
    async def _query_with_retry(
        self,
        query_embedding: List[float],
        n_results: int,
        filter_dict: Optional[Dict]
    ) -> Dict[str, Any]:
        """Query with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Run in thread pool
                results = await asyncio.to_thread(
                    self.collection.query,
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filter_dict,
                    include=['metadatas', 'documents', 'distances']
                )
                
                # Flatten results
                return {
                    'ids': results['ids'][0] if results['ids'] else [],
                    'distances': results['distances'][0] if results['distances'] else [],
                    'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                    'documents': results['documents'][0] if results['documents'] else []
                }
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"Query attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def delete_document(self, doc_id: str):
        """
        Delete all embeddings for a document
        
        IMPROVED: Retry logic
        """
        if not self.is_initialized:
            await self.initialize()
        
        for attempt in range(self.max_retries):
            try:
                # Get all IDs (in thread pool)
                results = await asyncio.to_thread(
                    self.collection.get,
                    where={"doc_id": doc_id},
                    include=[]
                )
                
                if results['ids']:
                    # Delete (in thread pool)
                    await asyncio.to_thread(
                        self.collection.delete,
                        ids=results['ids']
                    )
                    logger.info(f"✅ Deleted {len(results['ids'])} embeddings for doc {doc_id}")
                else:
                    logger.warning(f"No embeddings found for doc {doc_id}")
                
                return  # Success
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ Failed to delete document {doc_id}: {e}")
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"Delete attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def delete_all_documents(self):
        """
        Delete all documents from the collection
        
        IMPROVED: Proper async
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Delete and recreate collection (in thread pool)
            await asyncio.to_thread(
                self.client.delete_collection,
                name=settings.CHROMA_COLLECTION_NAME
            )
            
            self.collection = await asyncio.to_thread(
                self.client.create_collection,
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "Multi-modal RAG embeddings"}
            )
            
            # Clear cache
            if self.cache:
                self.cache.clear()
            
            logger.info("✅ All documents deleted from ChromaDB")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete all documents: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        
        IMPROVED: More detailed stats including cache
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            count = await asyncio.to_thread(self.collection.count)
            
            stats = {
                "name": settings.CHROMA_COLLECTION_NAME,
                "count": count,
                "model": settings.SENTENCE_TRANSFORMER_MODEL,
                "device": self.device,
                "embedding_dim": self.get_embedding_dimension(),
                "batch_size": self.batch_size,
                "stats": {
                    "total_embeddings_created": self.stats['total_embeddings_created'],
                    "total_items_stored": self.stats['total_items_stored'],
                    "total_queries": self.stats['total_queries'],
                }
            }
            
            # Add cache stats
            if self.cache:
                stats["cache"] = self.cache.get_stats()
            
            return stats
        
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {
                "name": settings.CHROMA_COLLECTION_NAME,
                "count": 0,
                "error": str(e)
            }
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.text_model:
            return self.text_model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key from text
        
        Uses MD5 hash for efficiency
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def warmup_cache(self, common_queries: List[str]):
        """
        Pre-populate cache with common queries
        
        NEW: Cache warming for frequently used queries
        
        Args:
            common_queries: List of common query texts
        """
        if not self.cache:
            logger.warning("Cache not enabled, skipping warmup")
            return
        
        logger.info(f"Warming up cache with {len(common_queries)} queries...")
        
        # Embed all queries (will be cached)
        await self.embed_texts_batch(common_queries, show_progress=False)
        
        logger.info(f"✅ Cache warmed up with {len(common_queries)} queries")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            **self.cache.get_stats()
        }
    
    async def clear_cache(self):
        """Clear embedding cache"""
        if self.cache:
            self.cache.clear()
            logger.info("✅ Embedding cache cleared")
        else:
            logger.warning("Cache not enabled")
    
    # ====================== Advanced Features ======================
    
    async def batch_query(
        self,
        queries: List[str],
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch query multiple queries efficiently
        
        NEW: Process multiple queries in parallel
        
        Args:
            queries: List of query texts
            n_results: Number of results per query
            filter_dict: Optional metadata filter
        
        Returns:
            List of result dictionaries
        """
        if not queries:
            return []
        
        logger.info(f"Batch querying {len(queries)} queries...")
        
        # Create tasks for parallel execution
        tasks = [
            self.query(query, n_results, filter_dict)
            for query in queries
        ]
        
        # Execute all queries in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {idx} failed: {result}")
                final_results.append({
                    'ids': [],
                    'distances': [],
                    'metadatas': [],
                    'documents': [],
                    'error': str(result)
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def rerank_results(
        self,
        query_text: str,
        results: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Re-rank results using cross-encoder (optional enhancement)
        
        NEW: Improve ranking quality with re-ranking
        
        Note: Requires cross-encoder model (not implemented yet)
        """
        # Placeholder for future cross-encoder integration
        logger.warning("Re-ranking not implemented yet")
        
        if top_k and top_k < len(results['ids']):
            # Simple truncation for now
            return {
                'ids': results['ids'][:top_k],
                'distances': results['distances'][:top_k],
                'metadatas': results['metadatas'][:top_k],
                'documents': results['documents'][:top_k],
            }
        
        return results
    
    async def get_similar_documents(
        self,
        doc_id: str,
        item_id: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Find documents similar to a specific item
        
        NEW: Document similarity search
        
        Args:
            doc_id: Source document ID
            item_id: Source item ID
            n_results: Number of results
        
        Returns:
            Similar documents
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get the source item
            source_id = f"{doc_id}_{item_id}"
            
            source_data = await asyncio.to_thread(
                self.collection.get,
                ids=[source_id],
                include=['embeddings', 'documents']
            )
            
            if not source_data['ids']:
                raise ValueError(f"Item not found: {source_id}")
            
            # Get the embedding
            source_embedding = source_data['embeddings'][0]
            
            # Query for similar items
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[source_embedding],
                n_results=n_results + 1,  # +1 to exclude self
                include=['metadatas', 'documents', 'distances']
            )
            
            # Remove the source item itself
            filtered_results = {
                'ids': [],
                'distances': [],
                'metadatas': [],
                'documents': []
            }
            
            for i in range(len(results['ids'][0])):
                if results['ids'][0][i] != source_id:
                    filtered_results['ids'].append(results['ids'][0][i])
                    filtered_results['distances'].append(results['distances'][0][i])
                    filtered_results['metadatas'].append(results['metadatas'][0][i])
                    filtered_results['documents'].append(results['documents'][0][i])
            
            # Limit to n_results
            for key in filtered_results:
                filtered_results[key] = filtered_results[key][:n_results]
            
            return filtered_results
        
        except Exception as e:
            logger.error(f"❌ Failed to find similar documents: {e}")
            raise

