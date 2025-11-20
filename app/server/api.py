
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import structlog
from datetime import datetime

from config import settings
from app.server.auth import verify_token
from app.utils.parser import DocumentParser
from app.utils.summarizer import ContentSummarizer
from app.utils.embedder import EmbeddingManager
from app.utils.retriever import MultiVectorRetriever
from app.models.ollama_adapter import OllamaAdapter

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Multi-modal RAG system"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

parser = DocumentParser()

llm_adapter = OllamaAdapter(
    model_name=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL
)
logger.info(f"Using Ollama LLM: {settings.OLLAMA_MODEL}")

mllm_adapter = OllamaAdapter(
    model_name=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL
)
logger.info("Using Ollama for multimodal")

summarizer = ContentSummarizer(mllm_adapter)
embedder = EmbeddingManager()
retriever = MultiVectorRetriever()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_multimodal: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    processing_time: float

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_processed: Dict[str, int]
    message: str

@app.get("/", include_in_schema=False)
async def web_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", include_in_schema=False)
async def admin_interface(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    credentials: dict = Depends(verify_token)
):
    start_time = datetime.utcnow()

    try:
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        logger.info(f"Processing upload: {file.filename} (doc_id: {doc_id})")

        content = await file.read()
        parsed_data = await parser.parse_document(content, file.filename, file.content_type)

        all_summaries = []

        for idx, chunk in enumerate(parsed_data['text_chunks']):
            all_summaries.append({
                'id': f"text_{idx}",
                'summary': chunk[:500],
                'raw': chunk,
                'type': 'text'
            })

        for table in parsed_data['tables']:
            all_summaries.append({
                'id': table['id'],
                'summary': table['content'][:500],
                'raw': table['content'],
                'type': 'table'
            })

        for img in parsed_data['images']:
            all_summaries.append({
                'id': img['id'],
                'summary': f"Image: {img['id']}",
                'raw': img['base64'],
                'path': img['path'],
                'type': 'image'
            })

        counts = await embedder.embed_and_store(all_summaries, doc_id)
        await retriever.store_raw_documents(doc_id, all_summaries, file.filename)

        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Processed {file.filename} in {processing_time:.2f}s")

        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunks_processed=counts,
            message=f"Document processed in {processing_time:.2f}s"
        )

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    credentials: dict = Depends(verify_token)
):
    start_time = datetime.utcnow()

    try:
        logger.info(f"Processing query: {request.query[:100]}...")

        search_results = await embedder.query(request.query, n_results=request.top_k)

        if not search_results['ids']:
            return QueryResponse(
                answer="No documents found. Please upload documents first.",
                sources=[],
                processing_time=0.0
            )

        raw_docs = await retriever.retrieve_raw_documents(search_results['ids'])

        text_context = "\n\n".join(raw_docs['text_chunks'])
        table_context = raw_docs['table_chunks']

        full_context = text_context
        if table_context:
            full_context += "\n\nTables:\n" + "\n\n".join(table_context)

        prompt = (
            f"Context:\n{full_context}\n\n"
            f"Question: {request.query}\n\n"
            f"Provide an answer based on the context above."
        )

        answer = await llm_adapter.generate_text(prompt)

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        sources = []
        for i, (doc_id, distance) in enumerate(zip(search_results['ids'], search_results['distances'])):
            sources.append({
                'rank': i + 1,
                'doc_id': doc_id,
                'relevance_score': float(1 - distance),
                'type': search_results['metadatas'][i]['type']
            })

        logger.info(f"Query processed in {processing_time:.2f}s")

        return QueryResponse(
            answer=answer,
            sources=sources,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(credentials: dict = Depends(verify_token)):
    try:
        pattern = "doc_meta:*"
        keys = retriever.redis_client.keys(pattern)
        
        docs = []
        for key in keys:
            data = retriever.redis_client.get(key)
            if data:
                import json
                doc_info = json.loads(data)
                docs.append(doc_info)
        
        docs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{doc_id}")
async def delete_document(
    doc_id: str,
    credentials: dict = Depends(verify_token)
):
    try:
        embedder.delete_document(doc_id)
        retriever.delete_document(doc_id)
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "chroma": "operational",
            "redis": "operational",
            "llm": "operational"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
