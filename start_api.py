"""
Startup script for the FastAPI chatbot server
Run this script to start the chatbot API server
"""

import uvicorn
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    print("Starting B2B Lead Scoring Chatbot API...")
    print("API will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "api.chatbot_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

