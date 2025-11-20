"""
Multi-modal RAG System
A production-ready system for retrieval-augmented generation with multimodal support.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Multi-modal RAG System with FastAPI"

# Lazy import to avoid circular dependencies and early failures
def get_app():
    """Lazy load the FastAPI application."""
    from .server.api import app
    return app

# Export for convenience
try:
    from .server.api import app
    __all__ = ["app", "get_app", "__version__"]
except ImportError as e:
    # If server module isn't ready yet, only export get_app
    __all__ = ["get_app", "__version__"]
    import warnings
    warnings.warn(
        f"Could not import app directly: {e}. "
        "Use get_app() function instead.",
        ImportWarning
    )
