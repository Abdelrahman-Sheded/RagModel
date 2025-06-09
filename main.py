# main.py
from fastapi import FastAPI, HTTPException
import os
import sys

# Add the project root directory to Python's module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_db import initialize_system
from src.chat import interactive_chat
from config import ensure_model_files

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "API is running", "message": "Welcome to the RAG API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat_endpoint(message: str):
    try:
        # Ensure model files are available
        ensure_model_files()
        
        # Initialize the system
        cv_dir = "images"
        job_desc_path = "junior_devops_requirements.pdf"
        faiss_index, metadata = initialize_system(cv_dir)
        
        # Process the chat
        response = interactive_chat(faiss_index, metadata, job_desc_path, message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)