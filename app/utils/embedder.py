import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import structlog
from config import settings

logger = structlog.get_logger()

class EmbeddingManager:
    """
    Manages embeddings using:
    - Text: SentenceTransformers or OpenAI embeddings
    - Images: CLIP embeddings
    - Stores in ChromaDB with multi-vector retrieval support
    """
    
    def __init__(self):
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding models
        self.text_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
        
        logger.info("Initialized EmbeddingManager with ChromaDB")
    
    async def embed_and_store(
        self,
        summaries: List[Dict[str, Any]],
        doc_id: str
    ) -> Dict[str, int]:
        """
        Embed summaries and store in ChromaDB.
        Returns counts of embedded items by type.
        """
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        for item in summaries:
            # Generate embedding from summary
            embedding = self.text_model.encode(item['summary']).tolist()
            
            embeddings.append(embedding)
            documents.append(item['summary'])  # Store summary in ChromaDB
            metadatas.append({
                'doc_id': doc_id,
                'item_id': item['id'],
                'type': item['type'],
                'has_raw': 'true'
            })
            ids.append(f"{doc_id}_{item['id']}")
        
        # Add to ChromaDB
        if embeddings:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        counts = {
            'text': sum(1 for s in summaries if s['type'] == 'text'),
            'table': sum(1 for s in summaries if s['type'] == 'table'),
            'image': sum(1 for s in summaries if s['type'] == 'image')
        }
        
        logger.info(f"Stored {len(embeddings)} embeddings for doc {doc_id}: {counts}")
        return counts
    
    async def query(
        self,
        query_text: str,
        n_results: int = None
    ) -> Dict[str, Any]:
        """
        Query ChromaDB with text and return results with metadata.
        """
        n_results = n_results or settings.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.text_model.encode(query_text).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        
        logger.info(f"Retrieved {len(results['ids'][0])} results for query")
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def delete_document(self, doc_id: str):
        """Delete all embeddings for a document."""
        results = self.collection.get(where={"doc_id": doc_id})
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} embeddings for doc {doc_id}")
