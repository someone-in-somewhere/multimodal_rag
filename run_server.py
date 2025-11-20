#!/usr/bin/env python3
# Script to run the Multi-modal RAG server.

import uvicorn
from config import settings

if __name__ == "__main__":
    print(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    print(f"Server running on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    
    uvicorn.run(
        "app.server.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )