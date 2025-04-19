from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.vector_db import initialize_system
from src.ranking import rank_cvs
from src.chat import generate_response
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
# uvicorn api.test_api:app --reload --port 8000
app = FastAPI() 

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
cv_dir = "images"
job_desc_path = "junior_devops_requirements.pdf"
faiss_index, metadata = initialize_system(cv_dir)
ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)  # Should return all candidates

class ChatRequest(BaseModel):
    message: str

class CandidateRequest(BaseModel):
    top_n: int = 0  # 0 means all candidates

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "cv_count": len(metadata),
        "ranking_ready": bool(ranked_cvs),
        "total_candidates": len(ranked_cvs)
    }

@app.post("/chat")
def chat_api(request: ChatRequest):
    response_text = generate_response(
        user_prompt=request.message,
        faiss_index=faiss_index,
        metadata=metadata,
        job_desc_path=job_desc_path
    )
    return {"response": response_text}

@app.post("/candidates")
def get_candidates(request: CandidateRequest):
    if not ranked_cvs:
        return {"error": "Ranking system not initialized"}
    
    # Handle "all candidates" request
    if request.top_n <= 0:
        top_n = len(ranked_cvs)
    else:
        top_n = min(request.top_n, len(ranked_cvs))
    
    candidates = []
    for idx, cv in enumerate(ranked_cvs[:top_n]):
        candidates.append({
            "id": idx,
            "filename": cv["filename"],
            "similarity": round(cv["similarity"], 4),
            "contact": cv.get("contact", {"email": "N/A", "phone": "N/A"}),
            "summary": cv.get("cleaned_text", "")[:500] + "..."
        })
    
    return {
        "candidates": candidates,
        "total_candidates": len(ranked_cvs)
    }

@app.get("/candidates/{candidate_id}")
def get_candidate_details(candidate_id: int):
    if candidate_id < 0 or candidate_id >= len(ranked_cvs):
        return {"error": "Invalid candidate ID"}
    
    cv = ranked_cvs[candidate_id]
    return {
        "id": candidate_id,
        "filename": cv["filename"],
        "similarity": cv["similarity"],
        "contact": cv.get("contact", {"email": "N/A", "phone": "N/A"}),
        "experience": cv.get("experience", "Not specified"),
        "skills": cv.get("skills", []),
        "full_text": cv.get("raw_text", "")
    }

# Add these to existing code
@app.post("/upload-cv")
async def upload_cv(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    file: UploadFile = File(...)
):
    global faiss_index, metadata, ranked_cvs
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Only PDF files are allowed"}
        )

    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("images")
        uploads_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_title = "".join(c if c.isalnum() else "_" for c in title)
        filename = f"{sanitized_title}_{timestamp}.pdf"
        file_path = uploads_dir / filename

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add CV to the system
        updated_index, updated_metadata, success, message = add_cv(
            str(file_path),
            faiss_index,
            metadata,
            filename  # Pass original filename
        )

        if success:
            # Update global state
            faiss_index = updated_index
            metadata = updated_metadata
            
            # Schedule ranking update
            background_tasks.add_task(update_rankings)
            
            return {
                "status": "success",
                "message": "CV uploaded successfully",
                "filename": filename,
                "path": str(file_path)
            }
        else:
            # Clean up failed upload
            if file_path.exists():
                file_path.unlink()
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": message}
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error processing CV: {str(e)}"}
        )