"""
FastAPI Chatbot API for B2B Lead Scoring System
Provides REST API endpoint for the RAG-based chatbot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path to import RAG modules
sys.path.append(str(Path(__file__).parent.parent))

from rag.context_builder import ContextBuilder
from rag.query_engine import QueryEngine

app = FastAPI(
    title="B2B Lead Scoring Chatbot API",
    description="RAG-based chatbot API for lead scoring and ABM insights",
    version="1.0.0"
)

# CORS middleware to allow Streamlit to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str
    top_k: Optional[int] = 5


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    contexts_used: int
    sources: list


# Global variables for context builder and query engine
context_builder: Optional[ContextBuilder] = None
query_engine: Optional[QueryEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize context builder and query engine on startup"""
    global context_builder, query_engine
    
    try:
        # Initialize context builder
        print("Initializing context builder...")
        context_builder = ContextBuilder()
        collection = context_builder.get_collection()
        
        # Check if collection exists and has data
        try:
            count = collection.count()
            print(f"ChromaDB collection has {count} documents")
        except Exception as e:
            print(f"Warning: Could not count collection: {e}")
        
        # Initialize query engine (use template mode if no API key)
        print("Initializing query engine...")
        model_provider = "template"  # Change to "openai", "groq", or "gemini" if API keys are available
        query_engine = QueryEngine(collection, model_provider=model_provider)
        
        print("✅ Chatbot API initialized successfully")
    except Exception as e:
        import traceback
        print(f"❌ Error initializing chatbot: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        print("The API will run but may have limited functionality")
        # Don't fail completely - allow the API to start but mark components as unavailable


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - processes user queries and returns AI responses
    
    Args:
        request: ChatRequest with user query
        
    Returns:
        ChatResponse with AI-generated answer
    """
    if query_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Query engine not initialized. Please check server logs."
        )
    
    try:
        # Check if collection has data
        if context_builder is not None:
            collection = context_builder.get_collection()
            count = collection.count()
            if count == 0:
                return ChatResponse(
                    response="No data available. Please upload and process lead data in the Streamlit app first, then try again.",
                    contexts_used=0,
                    sources=[]
                )
        
        result = query_engine.query(request.query, top_k=request.top_k)
        return ChatResponse(
            response=result["response"],
            contexts_used=result["contexts_used"],
            sources=result["sources"]
        )
    except Exception as e:
        import traceback
        error_detail = f"Error processing query: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"API Error: {error_detail}")  # Log to server console
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/rebuild-context")
async def rebuild_context():
    """
    Rebuild the context from current data
    This endpoint should be called when new data is uploaded
    """
    if context_builder is None:
        raise HTTPException(
            status_code=503,
            detail="Context builder not initialized"
        )
    
    # Note: In a real implementation, you'd pass the dataframe here
    # For now, this is a placeholder
    return {"message": "Context rebuild endpoint. Implement data passing logic."}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "context_builder_ready": context_builder is not None,
        "query_engine_ready": query_engine is not None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "B2B Lead Scoring Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

