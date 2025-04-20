from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.vector_db import initialize_system
from src.ranking import rank_cvs
from src.chat import generate_response
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from src.cv_management import add_cv
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
    # Add proper error checking
    if faiss_index is None or not metadata:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy"}
        )
    
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

def update_rankings():
    """Update candidate rankings in the background"""
    global faiss_index, metadata, ranked_cvs
    try:
        if faiss_index and metadata:
            ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
    except Exception as e:
        print(f"Ranking update error: {str(e)}")

@app.post("/upload-cv")
async def upload_cv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    global faiss_index, metadata, ranked_cvs
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed"
            )

        # Validate file size (1MB limit)
        max_size = 1024 * 1024  # 1MB
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds 1MB limit"
            )
        file.file.seek(0)

        # Create upload directory if needed
        cv_dir.mkdir(exist_ok=True, parents=True)

        # Generate safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(file.filename).stem
        safe_filename = f"{original_filename}_{timestamp}.pdf"
        file_path = cv_dir / safe_filename

        # Save the file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add to vector database
        try:
            new_index, new_metadata = add_cv(
                str(file_path),
                faiss_index,
                metadata,
                title=title
            )
        except Exception as e:
            file_path.unlink(missing_ok=True)  # Clean up file
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing CV: {str(e)}"
            )

        # Update global state
        faiss_index = new_index
        metadata = new_metadata

        # Schedule background ranking update
        background_tasks.add_task(update_rankings)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "success",
                "message": "CV uploaded and processed successfully",
                "filename": safe_filename,
                "path": str(file_path)
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )