"""
FastAPI application and API endpoints
Main server module handling all HTTP requests
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.requests import Request

from config import settings
from app.models.llm_adapter import LLMAdapter
from app.models.mllm_adapter import MLLMAdapter
from app.utils.parser import DocumentParser
from app.utils.summarizer import ContentSummarizer
from app.utils.embedder import EmbeddingManager
from app.utils.retriever import MultiVectorRetriever
from app.server.auth import verify_api_key

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Global Components (will be initialized in lifespan)
# ============================================================================

parser: Optional[DocumentParser] = None
llm_adapter: Optional[LLMAdapter] = None
mllm_adapter: Optional[MLLMAdapter] = None
summarizer: Optional[ContentSummarizer] = None
embedder: Optional[EmbeddingManager] = None
retriever: Optional[MultiVectorRetriever] = None


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # ============ STARTUP ============
    logger.info("🚀 Starting Multi-modal RAG System...")
    
    global parser, llm_adapter, mllm_adapter, summarizer, embedder, retriever
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Document Parser
        parser = DocumentParser()
        logger.info("✅ DocumentParser initialized")
        
        # LLM Adapters
        llm_adapter = LLMAdapter()
        await llm_adapter.initialize()
        logger.info("✅ LLMAdapter initialized")
        
        mllm_adapter = MLLMAdapter()
        await mllm_adapter.initialize()
        logger.info("✅ MLLMAdapter initialized")
        
        # Content Summarizer
        summarizer = ContentSummarizer(mllm_adapter)
        logger.info("✅ ContentSummarizer initialized")
        
        # Embedding Manager
        embedder = EmbeddingManager()
        await embedder.initialize()
        logger.info("✅ EmbeddingManager initialized")
        
        # Multi-Vector Retriever
        retriever = MultiVectorRetriever()
        await retriever.initialize()
        logger.info("✅ MultiVectorRetriever initialized")
        
        logger.info("🎉 All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield  # Application is running
    
    # ============ SHUTDOWN ============
    logger.info("🛑 Shutting down Multi-modal RAG System...")
    
    try:
        # Cleanup resources
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
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Multi-modal RAG System",
    description="A FastAPI-based RAG system supporting text, tables, and images",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# CORS Middleware
# ============================================================================

if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ============================================================================
# Static Files & Templates
# ============================================================================

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="User query", min_length=1, max_length=2000)
    top_k: int = Field(5, description="Number of results to retrieve", ge=1, le=20)
    use_multimodal: bool = Field(False, description="Use multimodal generation")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    sources: List[dict]
    processing_time: float


class UploadResponse(BaseModel):
    """Response model for upload endpoint"""
    doc_id: str
    filename: str
    chunks_processed: dict
    message: str


class DocumentInfo(BaseModel):
    """Document information model"""
    doc_id: str
    filename: str
    chunks: dict
    timestamp: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    components: dict
    timestamp: str


# ============================================================================
# HTML Pages
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Admin panel for document management"""
    return templates.TemplateResponse("admin.html", {"request": request})


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns status of all components
    """
    from datetime import datetime
    
    components_health = {}
    
    # Check each component
    if llm_adapter:
        components_health["llm_adapter"] = await llm_adapter.health_check()
    
    if mllm_adapter:
        components_health["mllm_adapter"] = await mllm_adapter.health_check()
    
    if embedder:
        try:
            stats = await embedder.get_collection_stats()
            components_health["embedder"] = {
                "status": "healthy",
                "documents": stats.get("count", 0)
            }
        except Exception as e:
            components_health["embedder"] = {
                "status": "error",
                "error": str(e)
            }
    
    if retriever:
        try:
            components_health["retriever"] = {
                "status": "healthy" if retriever.redis_client.ping() else "unhealthy"
            }
        except Exception as e:
            components_health["retriever"] = {
                "status": "error",
                "error": str(e)
            }
    
    # Overall status
    all_healthy = all(
        comp.get("status") == "healthy" 
        for comp in components_health.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": components_health,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Upload and process a document
    
    Supports: PDF, DOCX, HTML, TXT, MD, Images
    
    Process:
    1. Parse document (extract text, tables, images)
    2. Summarize all content
    3. Create embeddings
    4. Store in ChromaDB + Redis
    
    Returns:
        Document ID and processing statistics
    """
    start_time = time.time()
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    # Check file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE}MB"
        )
    
    logger.info(f"📤 Processing upload: {file.filename} ({file_size_mb:.2f}MB)")
    
    try:
        # Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Step 1: Parse document
        logger.info(f"📄 Parsing document...")
        parsed_data = await parser.parse_document(
            content,
            file.filename,
            file.content_type
        )
        
        # Step 2: Summarize all content
        logger.info(f"✍️  Summarizing content...")
        all_summaries = []
        
        # Summarize text chunks
        for idx, chunk in enumerate(parsed_data['text_chunks']):
            try:
                summary = await summarizer.summarize(chunk, content_type="text")
                all_summaries.append({
                    'id': f"text_{idx}",
                    'summary': summary,
                    'raw': chunk,
                    'type': 'text'
                })
            except Exception as e:
                logger.error(f"Error summarizing text chunk {idx}: {e}")
        
        # Summarize tables
        for table in parsed_data['tables']:
            try:
                summary = await summarizer.summarize(
                    table['content'],
                    content_type="table"
                )
                all_summaries.append({
                    'id': table['id'],
                    'summary': summary,
                    'raw': table['content'],
                    'type': 'table'
                })
            except Exception as e:
                logger.error(f"Error summarizing table {table['id']}: {e}")
        
        # Summarize images
        for img in parsed_data['images']:
            try:
                summary = await summarizer.summarize_image(img['base64'])
                all_summaries.append({
                    'id': img['id'],
                    'summary': summary,
                    'raw': img['base64'],
                    'path': img['path'],
                    'type': 'image'
                })
            except Exception as e:
                logger.error(f"Error summarizing image {img['id']}: {e}")
        
        if not all_summaries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content could be extracted from the document"
            )
        
        # Step 3: Create embeddings and store
        logger.info(f"🔢 Creating embeddings...")
        counts = await embedder.embed_and_store(all_summaries, doc_id)
        
        # Step 4: Store raw documents in Redis
        logger.info(f"💾 Storing in Redis...")
        await retriever.store_raw_documents(doc_id, all_summaries, file.filename)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"✅ Document processed: {doc_id} in {processing_time:.2f}s "
            f"(text: {counts['text']}, tables: {counts['table']}, images: {counts['image']})"
        )
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks_processed": counts,
            "message": f"Document processed successfully in {processing_time:.2f}s"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Query documents and generate answer
    
    Process:
    1. Embed query
    2. Semantic search in ChromaDB
    3. Retrieve raw content from Redis
    4. Generate answer with LLM/MLLM
    
    Returns:
        Answer, sources, and processing time
    """
    start_time = time.time()
    
    logger.info(f"❓ Query: {request.query[:100]}...")
    
    try:
        # Step 1: Semantic search
        logger.info(f"🔍 Searching for relevant documents...")
        search_results = await embedder.query(request.query, n_results=request.top_k)
        
        if not search_results['ids']:
            logger.info("📭 No relevant documents found")
            return {
                "answer": "Xin lỗi, tôi không tìm thấy tài liệu nào liên quan đến câu hỏi của bạn. Vui lòng thử câu hỏi khác hoặc upload thêm tài liệu.",
                "sources": [],
                "processing_time": time.time() - start_time
            }
        
        logger.info(f"📚 Found {len(search_results['ids'])} relevant chunks")
        
        # Step 2: Retrieve raw documents
        logger.info(f"💾 Retrieving full content...")
        raw_docs = await retriever.retrieve_raw_documents(search_results['ids'])
        
        # Step 3: Build context
        text_context = "\n\n".join(raw_docs['text_chunks']) if raw_docs['text_chunks'] else ""
        table_context = raw_docs['table_chunks']
        image_context = raw_docs['image_chunks']
        
        # Step 4: Generate answer
        logger.info(f"🤖 Generating answer...")
        
        if request.use_multimodal and (image_context or table_context):
            # Use multimodal generation
            answer = await mllm_adapter.generate_multimodal(
                text=text_context,
                tables=table_context,
                images=image_context,
                max_tokens=1000,
                temperature=0.7
            )
        else:
            # Use text-only generation
            full_context = text_context
            
            if table_context:
                full_context += "\n\nBảng dữ liệu:\n" + "\n\n".join(table_context)
            
            prompt = f"""Dựa trên context dưới đây, hãy trả lời câu hỏi một cách chi tiết và chính xác.

Context:
{full_context}

Câu hỏi: {request.query}

Trả lời chi tiết dựa trên context trên:"""
            
            answer = await llm_adapter.generate_text(
                prompt,
                max_tokens=1000,
                temperature=0.7
            )
        
        # Build sources list
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
                'relevance_score': relevance_score,
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
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/documents")
async def list_documents(
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    List all documents
    
    Returns:
        List of documents with metadata
    """
    try:
        documents = await retriever.list_all_documents()
        
        return {
            "total": len(documents),
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Delete a specific document
    
    Args:
        doc_id: Document ID to delete
    
    Returns:
        Success message
    """
    try:
        logger.info(f"🗑️  Deleting document: {doc_id}")
        
        # Delete from ChromaDB
        await embedder.delete_document(doc_id)
        
        # Delete from Redis
        await retriever.delete_document(doc_id)
        
        logger.info(f"✅ Document deleted: {doc_id}")
        
        return {
            "message": f"Document {doc_id} deleted successfully"
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to delete document {doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.delete("/documents")
async def delete_all_documents(
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Delete all documents
    
    WARNING: This will delete ALL documents from the system
    
    Returns:
        Success message with count
    """
    try:
        logger.info("🗑️  Deleting all documents...")
        
        # Get all documents first
        documents = await retriever.list_all_documents()
        count = len(documents)
        
        # Delete all from ChromaDB
        await embedder.delete_all_documents()
        
        # Delete all from Redis
        await retriever.delete_all_documents()
        
        logger.info(f"✅ All documents deleted (count: {count})")
        
        return {
            "message": f"All {count} documents deleted successfully",
            "count": count
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to delete all documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete all documents: {str(e)}"
        )


@app.get("/stats")
async def get_stats(
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get system statistics
    
    Returns:
        Statistics about documents, chunks, etc.
    """
    try:
        # Get ChromaDB stats
        collection_stats = await embedder.get_collection_stats()
        
        # Get Redis stats
        documents = await retriever.list_all_documents()
        
        # Calculate totals
        total_text = sum(doc.get('chunks', {}).get('text', 0) for doc in documents)
        total_tables = sum(doc.get('chunks', {}).get('table', 0) for doc in documents)
        total_images = sum(doc.get('chunks', {}).get('image', 0) for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_chunks": collection_stats.get('count', 0),
            "chunks_by_type": {
                "text": total_text,
                "table": total_tables,
                "image": total_images
            },
            "collection_name": collection_stats.get('name', 'unknown')
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.LOG_LEVEL == "DEBUG" else "An error occurred"
        }
    )


# ============================================================================
# Startup Message
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Print startup message"""
    logger.info("="*70)
    logger.info("🎉 Multi-modal RAG System is ready!")
    logger.info(f"📍 Server: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"📖 Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info(f"🔑 API Key: {settings.API_KEY[:10]}...")
    logger.info("="*70)
