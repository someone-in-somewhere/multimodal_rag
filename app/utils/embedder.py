"""
Embedding Manager
Handles text embeddings and vector storage in ChromaDB
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embeddings and vector storage
    
    Features:
    - Create embeddings from text using sentence-transformers
    - Store embeddings in ChromaDB
    - Semantic search
    - Document management
    """
    
    def __init__(self):
        """Initialize the embedding manager"""
        self.client = None
        self.collection = None
        self.text_model = None
        self.is_initialized = False
        
        logger.info("EmbeddingManager created (not initialized)")
    
    async def initialize(self):
        """Initialize ChromaDB and embedding model"""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing EmbeddingManager...")
            
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
            
            # Initialize sentence transformer model
            logger.info(f"Loading embedding model: {settings.SENTENCE_TRANSFORMER_MODEL}")
            self.text_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
            logger.info("✅ Embedding model loaded")
            
            self.is_initialized = True
            logger.info("✅ EmbeddingManager initialized successfully")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize EmbeddingManager: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        self.client = None
        self.collection = None
        self.text_model = None
        self.is_initialized = False
        
        logger.info("🧹 EmbeddingManager cleaned up")
    
    async def embed_and_store(
        self,
        summaries: List[Dict[str, Any]],
        doc_id: str
    ) -> Dict[str, int]:
        """
        Create embeddings and store in ChromaDB
        
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
        
        logger.info(f"Creating embeddings for {len(summaries)} items...")
        
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        counts = {"text": 0, "table": 0, "image": 0}
        
        for item in summaries:
            try:
                # Create embedding from summary
                embedding = self.text_model.encode(item['summary']).tolist()
                
                embeddings.append(embedding)
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
            
            except Exception as e:
                logger.error(f"Failed to embed item {item.get('id', 'unknown')}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings created")
        
        # Store in ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(
                f"✅ Stored {len(embeddings)} embeddings "
                f"(text: {counts['text']}, table: {counts['table']}, image: {counts['image']})"
            )
            
            return counts
        
        except Exception as e:
            logger.error(f"❌ Failed to store embeddings: {e}")
            raise
    
    async def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents
        
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
            # Create query embedding
            query_embedding = self.text_model.encode(query_text).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Flatten results (ChromaDB returns nested lists)
            return {
                'ids': results['ids'][0] if results['ids'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'documents': results['documents'][0] if results['documents'] else []
            }
        
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise
    
    async def delete_document(self, doc_id: str):
        """
        Delete all embeddings for a document
        
        Args:
            doc_id: Document ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=[]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"✅ Deleted {len(results['ids'])} embeddings for doc {doc_id}")
            else:
                logger.warning(f"No embeddings found for doc {doc_id}")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete document {doc_id}: {e}")
            raise
    
    async def delete_all_documents(self):
        """Delete all documents from the collection"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
            
            self.collection = self.client.create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "Multi-modal RAG embeddings"}
            )
            
            logger.info("✅ All documents deleted from ChromaDB")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete all documents: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Returns:
            Dictionary with collection stats
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            count = self.collection.count()
            
            return {
                "name": settings.CHROMA_COLLECTION_NAME,
                "count": count,
                "model": settings.SENTENCE_TRANSFORMER_MODEL
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {
                "name": settings.CHROMA_COLLECTION_NAME,
                "count": 0,
                "error": str(e)
            }
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings
        
        Returns:
            Embedding dimension
        """
        if self.text_model:
            return self.text_model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2
