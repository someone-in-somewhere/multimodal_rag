
import redis
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from config import settings

logger = structlog.get_logger()

class MultiVectorRetriever:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
            decode_responses=False
        )
        logger.info("Initialized MultiVectorRetriever with Redis")
    
    async def store_raw_documents(
        self,
        doc_id: str,
        summaries: List[Dict[str, Any]],
        filename: str = None
    ):
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
            
            self.redis_client.set(key, json.dumps(data))
        
        chunks_count = {
            'text': sum(1 for s in summaries if s['type'] == 'text'),
            'table': sum(1 for s in summaries if s['type'] == 'table'),
            'image': sum(1 for s in summaries if s['type'] == 'image')
        }
        
        meta_key = f"doc_meta:{doc_id}"
        self.redis_client.set(meta_key, json.dumps({
            'doc_id': doc_id,
            'filename': filename or doc_id,
            'item_count': len(summaries),
            'chunks': chunks_count,
            'timestamp': datetime.utcnow().isoformat()
        }))
        
        logger.info(f"Stored {len(summaries)} docs in Redis for {doc_id}")
    
    async def retrieve_raw_documents(
        self,
        ids: List[str]
    ) -> Dict[str, List[Any]]:
        text_chunks = []
        table_chunks = []
        images = []
        
        for item_id in ids:
            key = f"doc:{item_id}"
            data = self.redis_client.get(key)
            
            if data:
                item = json.loads(data)
                
                if item['type'] == 'text':
                    text_chunks.append(item['raw'])
                elif item['type'] == 'table':
                    table_chunks.append(item['raw'])
                elif item['type'] == 'image':
                    images.append(item['raw'])
        
        logger.info(f"Retrieved {len(text_chunks)} text, {len(table_chunks)} tables, {len(images)} images")
        
        return {
            'text_chunks': text_chunks,
            'table_chunks': table_chunks,
            'images': images
        }
    
    def delete_document(self, doc_id: str):
        pattern = f"doc:{doc_id}:*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            self.redis_client.delete(*keys)
            self.redis_client.delete(f"doc_meta:{doc_id}")
            logger.info(f"Deleted {len(keys)} docs from Redis for {doc_id}")
