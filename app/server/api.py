"""
api_no_auth.py - API Without Authentication
============================================
For personal projects running locally only

WARNINGS:
⚠️ DO NOT expose to internet without auth!
⚠️ Anyone can upload/delete documents
⚠️ No rate limiting or protection

Safe when:
✅ Running on localhost only
✅ Personal/development use
✅ Behind firewall/VPN
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.requests import Request

from config import settings

# Import V2/V3 components
from app.models.llm_adapter import LLMAdapter
from app.models.mllm_adapter import MLLMAdapter
from app.utils.parser_v3 import DocumentParser
from app.utils.summarizer_v2 import ContentSummarizer
from app.utils.embedder_v2 import EmbeddingManager
from app.utils.retriever_v2 import MultiVectorRetriever

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Global Components
# ============================================================================

parser: Optional[DocumentParser] = None
llm_adapter: Optional[LLMAdapter] = None
mllm_adapter: Optional[MLLMAdapter] = None
summarizer: Optional[ContentSummarizer] = None
embedder: Optional[EmbeddingManager] = None
retriever: Optional[MultiVectorRetriever] = None


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    logger.info("🚀 Starting RAG System (NO AUTH - LOCAL ONLY)...")
    logger.warning("⚠️ Authentication DISABLED - Do not expose to internet!")
    
    global parser, llm_adapter, mllm_adapter, summarizer, embedder, retriever
    
    try:
        # Initialize V2 components
        parser = DocumentParser(
            extract_pdf_embedded_images=True,
            convert_pdf_pages_to_images=True,
        )
        
        llm_adapter = LLMAdapter()
        await llm_adapter.initialize()
        
        mllm_adapter = MLLMAdapter()
        await mllm_adapter.initialize()
        
        summarizer = ContentSummarizer(
            mllm_adapter,
            enable_cache=True,
            max_concurrent=5,
        )
        
        embedder = EmbeddingManager(
            batch_size=32,
            enable_cache=True,
        )
        await embedder.initialize()
        
        retriever = MultiVectorRetriever(
            enable_compression=True,
            enable_cache=True,
        )
        await retriever.initialize()
        
        logger.info("🎉 All components initialized!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    
    try:
        if llm_adapter:
            await llm_adapter.cleanup()
        if mllm_adapter:
            await mllm_adapter.cleanup()
        if embedder:
            await embedder.cleanup()
        if retriever:
            await retriever.cleanup()
        
        logger.info("✅ Cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Multi-modal RAG System (No Auth)",
    description="Personal RAG system - Localhost only!",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================================================
# Models
# ============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    use_multimodal: bool = Field(False)


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    processing_time: float


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    chunks_processed: dict
    message: str
    processing_time: float


# ============================================================================
# HTML Pages
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Admin panel"""
    return templates.TemplateResponse("admin.html", {"request": request})


# ============================================================================
# API Endpoints (NO AUTH REQUIRED!)
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check"""
    components_health = {}
    
    try:
        if llm_adapter:
            components_health["llm_adapter"] = await llm_adapter.health_check()
        
        if mllm_adapter:
            components_health["mllm_adapter"] = await mllm_adapter.health_check()
        
        if embedder:
            stats = await embedder.get_collection_stats()
            components_health["embedder"] = {
                "status": "healthy",
                "documents": stats.get("count", 0)
            }
        
        if retriever:
            retriever_health = await retriever.health_check()
            components_health["retriever"] = retriever_health
        
        all_healthy = all(
            comp.get("status") == "healthy" or comp.get("healthy") == True
            for comp in components_health.values()
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components_health,
            "timestamp": datetime.utcnow().isoformat(),
            "auth": "disabled"  # Indicates no auth
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload document - NO AUTH REQUIRED
    
    ⚠️ Anyone can upload!
    """
    start_time = time.time()
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE}MB"
        )
    
    logger.info(f"📤 Upload: {file.filename} ({file_size_mb:.2f}MB)")
    
    try:
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Parse
        parsed_data = await parser.parse_document(
            content,
            file.filename,
            file.content_type,
            doc_id=doc_id
        )
        
        doc_type = parsed_data.get('doc_type', 'unknown')
        
        # Summarize (parallel)
        all_summaries = await summarizer.summarize_parsed_document(
            parsed_data,
            max_length=300,
            show_progress=True
        )
        
        if not all_summaries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content extracted"
            )
        
        # Embed (batch)
        counts = await embedder.embed_and_store(all_summaries, doc_id)
        
        # Store (compressed)
        await retriever.store_raw_documents(doc_id, all_summaries, file.filename)
        
        total_time = time.time() - start_time
        
        logger.info(f"✅ Processed: {doc_id} in {total_time:.2f}s")
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "doc_type": doc_type,
            "chunks_processed": counts,
            "message": f"Processed in {total_time:.2f}s",
            "processing_time": total_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents - NO AUTH REQUIRED
    
    ⚠️ Anyone can query!
    """
    start_time = time.time()
    
    logger.info(f"❓ Query: {request.query[:100]}...")
    
    try:
        # Search
        search_results = await embedder.query(request.query, n_results=request.top_k)
        
        if not search_results['ids']:
            return {
                "answer": "Không tìm thấy tài liệu liên quan. Vui lòng upload tài liệu hoặc thử câu hỏi khác.",
                "sources": [],
                "processing_time": time.time() - start_time
            }
        
        # Retrieve
        raw_docs = await retriever.retrieve_raw_documents(search_results['ids'])
        
        # Build context
        text_context = "\n\n".join(raw_docs['text_chunks']) if raw_docs['text_chunks'] else ""
        table_context = raw_docs['table_chunks']
        image_context = raw_docs['image_chunks']
        
        # Generate answer
        if request.use_multimodal and (image_context or table_context):
            answer = await mllm_adapter.generate_multimodal(
                text=text_context,
                tables=table_context,
                images=image_context,
                max_tokens=1000,
                temperature=0.7
            )
        else:
            full_context = text_context
            
            if table_context:
                full_context += "\n\nBảng:\n" + "\n\n".join(table_context)
            
            prompt = f"""Context:
{full_context}

Câu hỏi: {request.query}

Trả lời:"""
            
            answer = await llm_adapter.generate_text(
                prompt,
                max_tokens=1000,
                temperature=0.7
            )
        
        # Sources
        sources = []
        for i, (doc_id, distance, metadata) in enumerate(zip(
            search_results['ids'],
            search_results['distances'],
            search_results['metadatas']
        )):
            relevance_score = float(1.0 - min(distance, 1.0))
            sources.append({
                'rank': i + 1,
                'doc_id': doc_id,
                'relevance_score': round(relevance_score, 3),
                'type': metadata.get('type', 'unknown')
            })
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Query completed in {processing_time:.2f}s")
        
        return {
            "answer": answer,
            "sources": sources,
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"❌ Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/documents")
async def list_documents():
    """List all documents - NO AUTH"""
    try:
        documents = await retriever.list_all_documents()
        return {
            "total": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document - NO AUTH"""
    try:
        await embedder.delete_document(doc_id)
        await retriever.delete_document(doc_id)
        
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/documents")
async def delete_all_documents():
    """Delete all - NO AUTH"""
    try:
        documents = await retriever.list_all_documents()
        count = len(documents)
        
        await embedder.delete_all_documents()
        await retriever.delete_all_documents()
        
        return {
            "message": f"Deleted {count} documents",
            "count": count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/stats")
async def get_stats():
    """System statistics"""
    try:
        embedder_stats = await embedder.get_stats()
        retriever_stats = await retriever.get_stats()
        summarizer_stats = await summarizer.get_stats()
        documents = await retriever.list_all_documents()
        
        total_text = sum(doc.get('chunks', {}).get('text', 0) for doc in documents)
        total_tables = sum(doc.get('chunks', {}).get('table', 0) for doc in documents)
        total_images = sum(doc.get('chunks', {}).get('image', 0) for doc in documents)
        
        return {
            "documents": {
                "total": len(documents),
                "total_chunks": embedder_stats.get('count', 0),
                "by_type": {
                    "text": total_text,
                    "table": total_tables,
                    "image": total_images
                }
            },
            "embedder": {
                "cache_hit_rate": embedder_stats.get('cache', {}).get('hit_rate', 0)
            },
            "retriever": {
                "compression_enabled": retriever_stats.get('features', {}).get('compression', False),
                "compression_savings": retriever_stats.get('compression', {}).get('savings_percent', 0)
            },
            "summarizer": {
                "total_summaries": summarizer_stats.get('total_summaries', 0),
                "cache_hit_rate": summarizer_stats.get('cache', {}).get('hit_rate', 0)
            },
            "auth": "disabled"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup message"""
    logger.info("="*70)
    logger.info("🎉 Multi-modal RAG System v2 is ready!")
    logger.warning("⚠️ AUTHENTICATION DISABLED - LOCALHOST ONLY!")
    logger.info(f"📍 Server: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"📖 Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info("="*70)
