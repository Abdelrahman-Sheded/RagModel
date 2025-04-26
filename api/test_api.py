# uvicorn api.test_api:app --reload --port 8000 hiring
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid
import tempfile
import json
from src.vector_db import initialize_system, save_data
from src.cv_management import add_cv, remove_cv_from_system
from src.ranking import rank_cvs
from src.text_processing import extract_text_from_pdf, clean_text
from src.chat import compare_candidates
from config import AZURE_CONFIG, DEPLOYMENT_NAME
from langchain_openai import AzureChatOpenAI
from pathlib import Path
import datetime
import io
import urllib.parse

# Initialize FastAPI app
app = FastAPI(title="CV Chatbot API", description="RESTful API for CV chatbot functionality")

# Add CORS middleware to allow cross-origin requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the system with CV data
cv_dir = "images"
job_desc_path = "junior_devops_requirements.pdf"

# Global variables to store system state
faiss_index, metadata = None, None
ranked_cvs = None

try:
    faiss_index, metadata = initialize_system(cv_dir)
    ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
except Exception as e:
    print(f"Error initializing system: {str(e)}")

# Initialize language model for chat
chat_model = AzureChatOpenAI(
    azure_endpoint=AZURE_CONFIG["azure_endpoint"],
    api_key=AZURE_CONFIG["api_key"],
    api_version=AZURE_CONFIG["api_version"],
    deployment_name=DEPLOYMENT_NAME,
    temperature=0.3
)

# Define request and response models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class CandidateComparisonRequest(BaseModel):
    candidate1_index: int
    candidate2_index: int

class JobRequirementsTextUpdate(BaseModel):
    title: str
    requirements_text: str

class JobApplication(BaseModel):
    job_id: str
    applicant_name: str
    email: str
    phone: str
    cover_letter: Optional[str] = None

# API endpoints
@app.get("/")
def read_root():
    return {"status": "API is running", "message": "Welcome to the CV Chatbot API"}

@app.get("/health")
def health_check():
    if faiss_index is None or metadata is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "System not initialized"})
    return {"status": "healthy", "cv_count": len(metadata)}

@app.get("/candidates")
def get_candidates(top_n: Optional[int] = 20):
    """Get the top N candidates for the job description"""
    global ranked_cvs
    
    if ranked_cvs is None:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    candidates = []
    for i, cv in enumerate(ranked_cvs[:min(top_n, len(ranked_cvs))]):
        candidates.append({
            "id": i,
            "filename": cv["filename"],
            "similarity": cv["similarity"],
            "contact": cv["contact"],
            "summary": cv["cleaned_text"][:6000] + "..." if len(cv["cleaned_text"]) > 1000 else cv["cleaned_text"]
        })
        
    return {"candidates": candidates}

@app.get("/candidates/{candidate_id}")
def get_candidate_details(candidate_id: int):
    """Get detailed information about a specific candidate"""
    global ranked_cvs
    
    if ranked_cvs is None:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    if candidate_id < 0 or candidate_id >= len(ranked_cvs):
        raise HTTPException(status_code=404, detail=f"Candidate with ID {candidate_id} not found")
        
    cv = ranked_cvs[candidate_id]
    return {
        "id": candidate_id,
        "filename": cv["filename"],
        "similarity": cv["similarity"],
        "contact": cv["contact"],
        "full_text": cv["raw_text"],
        "cleaned_text": cv["cleaned_text"]
    }

# Add this new endpoint to serve PDF files
@app.get("/pdf/{filename}")
def get_pdf(filename: str):
    """Serve PDF files directly"""
    pdf_path = os.path.join(cv_dir, filename)
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(pdf_path, media_type="application/pdf")

# Add a new endpoint to get job requirements/post
@app.get("/job-requirements")
def get_job_requirements():
    """Get the job post for candidate evaluation"""
    global job_desc_path
    
    try:
        if not os.path.exists(job_desc_path):
            return {"requirements": "Job description file not found"}
        
        job_text = extract_text_from_pdf(job_desc_path)
        cleaned_job_text = clean_text(job_text)
        
        return {
            "requirements": job_text,
            "cleaned_requirements": cleaned_job_text,
            "filename": os.path.basename(job_desc_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job requirements: {str(e)}")

@app.post("/job-requirements/upload-pdf", status_code=201)
async def upload_job_requirements_pdf(background_tasks: BackgroundTasks, 
                                     title: str = Form(...),
                                     file: UploadFile = File(...)):
    """Upload a new job post PDF file"""
    global job_desc_path, ranked_cvs, faiss_index, metadata
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Create jobs directory if it doesn't exist
        jobs_dir = Path("jobs")
        jobs_dir.mkdir(exist_ok=True)
        
        # Generate a filename with timestamp and title
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_title = "".join(c if c.isalnum() else "_" for c in title)
        filename = f"{sanitized_title}_{timestamp}.pdf"
        file_path = jobs_dir / filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Update the global job description path
        job_desc_path = str(file_path)
        
        # Update rankings in the background
        background_tasks.add_task(update_rankings)
        
        return {"status": "success", "message": f"Job requirements uploaded as {filename}", "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading job requirements: {str(e)}")

@app.post("/job-requirements/update-text", status_code=201)
async def update_job_requirements_text(background_tasks: BackgroundTasks, request: JobRequirementsTextUpdate):
    """Update job post from text input"""
    global job_desc_path, ranked_cvs
    
    try:
        # Create jobs directory if it doesn't exist
        jobs_dir = Path("jobs")
        jobs_dir.mkdir(exist_ok=True)
        
        # Generate a filename with timestamp and title
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_title = "".join(c if c.isalnum() else "_" for c in request.title)
        filename = f"{sanitized_title}_{timestamp}.pdf"
        file_path = jobs_dir / filename
        
        # Create a PDF from the text
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.cell(200, 10, txt=request.title, ln=True, align='C')
        
        # Add content with word wrapping
        pdf.multi_cell(0, 10, request.requirements_text)
        
        # Save the PDF
        pdf.output(str(file_path))
        
        # Update the global job description path
        job_desc_path = str(file_path)
        
        # Update rankings in the background
        background_tasks.add_task(update_rankings)
        
        return {"status": "success", "message": f"Job requirements created as {filename}", "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating job requirements: {str(e)}")

@app.get("/job-requirements/list")
def list_job_requirements():
    """List all available job post files"""
    jobs_dir = Path("jobs")
    if not jobs_dir.exists():
        jobs_dir.mkdir(exist_ok=True)
        return {"job_files": []}
    
    job_files = []
    for file in jobs_dir.glob("*.pdf"):
        filename = file.name
        slug = filename.lower().replace('.pdf', '').replace('_', '-')
        job_files.append({
            "filename": filename,
            "slug": slug,  # Add this line
            "path": str(file),
            "is_current": str(file) == job_desc_path,
            "created": datetime.datetime.fromtimestamp(file.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Also include the current job description if it's not in the jobs directory
    if job_desc_path and Path(job_desc_path).exists() and not any(job["path"] == job_desc_path for job in job_files):
        filename = Path(job_desc_path).name
        slug = filename.lower().replace('.pdf', '').replace('_', '-')
        job_files.append({
            "filename": filename,
            "slug": slug,  # Add this line
            "path": job_desc_path,
            "is_current": True,
            "created": datetime.datetime.fromtimestamp(Path(job_desc_path).stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return {"job_files": sorted(job_files, key=lambda x: x["is_current"], reverse=True)}

@app.post("/job-requirements/set-active/{path}")
def set_active_job_requirements(path: str, background_tasks: BackgroundTasks):
    """Set a specific job post file as active"""
    global job_desc_path
    
    # URL decode the path if needed
    import urllib.parse
    decoded_path = urllib.parse.unquote(path)
    
    file_path = Path(decoded_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Job requirements file not found: {path}")
    
    job_desc_path = str(file_path)
    
    # Update rankings in the background
    background_tasks.add_task(update_rankings)
    
    return {"status": "success", "message": f"Active job requirements set to {file_path.name}", "path": str(file_path)}

@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    """Chat with the AI assistant about candidates"""
    global ranked_cvs
    
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Check if we need to update the system message with candidate information
        system_message_index = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
        
        if system_message_index is None and ranked_cvs:
            # Get job requirements
            job_text = extract_text_from_pdf(job_desc_path)
            
            # If no system message exists but we have candidates, add one with both job and candidate info
            system_content = "You are an AI assistant that helps analyze CV candidates for job matching.\n\n"
            
            # Add job requirements
            system_content += "JOB REQUIREMENTS:\n"
            system_content += f"{job_text[:1000]}...\n\n"
            
            # Add explicit ranking information
            system_content += "TOP CANDIDATES (ranked from best to worst match):\n"
            system_content += "Candidates are numbered by their rank, where Candidate 1 is the best match, Candidate 2 is the second best, and so on.\n"
            system_content += "Lower numbered candidates should generally be considered better matches than higher numbered candidates.\n\n"
            
            for i, cv in enumerate(ranked_cvs[:10]):  # Include top 10 candidates
                system_content += f"Candidate {i+1} (Rank #{i+1}): {cv['filename']}\n"
                system_content += f"Similarity Score: {cv['similarity']:.2f}\n"
                system_content += f"Contact: {cv['contact']}\n"
                system_content += f"Summary: {cv['cleaned_text'][:6000]}...\n\n"
            
            system_content += "When comparing candidates, always remember that lower-ranked candidates (with smaller numbers) are better matches for the job requirements than higher-ranked candidates (with larger numbers).\n\n"
            
            # Add instruction to avoid "No information provided" responses
            system_content += "RESPONSE GUIDELINES:\n"
            system_content += "- Never respond with phrases like 'No information provided' or 'No information available'\n"
            system_content += "- If specific information isn't explicitly mentioned in a candidate's profile, make reasonable inferences based on their background, skills, and experience\n"
            system_content += "- If you're genuinely uncertain about a detail, say something like 'While not explicitly mentioned in their CV, based on their background in [relevant field], they likely...' or 'Their CV doesn't highlight this specific aspect, but...'\n"
            system_content += "- Always provide a helpful response that offers insights based on available information\n"
            system_content += "- If asked about very specific details not in the CV, acknowledge the limitation briefly and provide related information that IS available\n\n"
            
            # Add specific guidelines for comparing candidates with improved formatting
            system_content += "COMPARISON FORMAT GUIDELINES:\n"
            system_content += "- When comparing candidates, use a conversational paragraph style instead of formal sections\n" 
            system_content += "- DO NOT use a rigid structure with headings like 'Education:', 'Technical Skills:', 'Soft Skills:'\n"
            system_content += "- Instead, write a flowing analysis that naturally incorporates relevant information from both candidates\n"
            system_content += "- Begin with a brief introduction of both candidates and their overall fit for the position\n"
            system_content += "- Follow with direct comparisons of their relevant qualifications, focusing on job requirements\n"
            system_content += "- Conclude with your assessment of which candidate appears better suited and why\n"
            system_content += "- Focus only on information that is available and relevant to the job requirements\n"
            system_content += "- Never mention the absence of information - simply discuss what IS present in their profiles\n"
            
            messages.insert(0, {
                "role": "system", 
                "content": system_content
            })
        
        prompt_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        response = chat_model.invoke(prompt_str).content
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM: {str(e)}")

def update_rankings():
    """Update the ranked CVs after changes to the database"""
    global ranked_cvs, faiss_index, metadata
    try:
        ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
    except Exception as e:
        print(f"Error updating rankings: {str(e)}")

@app.delete("/remove-cv/{filename}")
def remove_cv_endpoint(filename: str, background_tasks: BackgroundTasks):
    """Remove a CV by filename"""
    global faiss_index, metadata, ranked_cvs
    
    if faiss_index is None or metadata is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # First check if the CV exists in metadata before attempting removal
    cv_exists = any(cv['filename'] == filename for cv in metadata)
    if not cv_exists:
        raise HTTPException(status_code=404, detail=f"CV {filename} not found")
        
    try:
        # Only try to remove if the CV exists
        updated_index, updated_metadata = remove_cv_from_system(filename, faiss_index, metadata)
        
        # Update globals
        faiss_index, metadata = updated_index, updated_metadata
        # Update rankings in the background
        background_tasks.add_task(update_rankings)
        return {"status": "success", "message": f"CV {filename} removed successfully"}
    except Exception as e:
        # Log the error for debugging
        print(f"Error in remove_cv_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing CV: {str(e)}")
    
@app.post("/upload-cv")
async def upload_cv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a new CV file with improved error handling and filename preservation"""
    global faiss_index, metadata, ranked_cvs
    
    if faiss_index is None or metadata is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    # Save the uploaded file to a temporary location
    temp_file_path = None
    try:
        # Create a temporary file with the same extension
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        # Process the CV with the updated function that returns status and error message
        # Pass the original filename to override the temp filename
        original_filename = file.filename
        updated_index, updated_metadata, success, message = add_cv(
            temp_file_path, 
            faiss_index, 
            metadata, 
            original_filename
        )
        
        if success:
            faiss_index, metadata = updated_index, updated_metadata
            # Update rankings in the background
            background_tasks.add_task(update_rankings)
            return {"status": "success", "message": f"CV {original_filename} uploaded successfully"}
        else:
            # Return the specific error message from the add_cv function
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        # Re-raise HTTPExceptions as is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading CV: {str(e)}")
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/apply/{job_id}", response_class=HTMLResponse)
async def get_job_application_page(job_id: str, request: Request):
    """Get the job application page for a specific job"""
    # In a real application, you would look up the job by ID in a database
    # For now, we'll just return the current job post
    try:
        job_info = get_job_requirements()
        job_title = job_info.get('filename', '').split('_')[0].replace('_', ' ')
        job_description = job_info.get('requirements', 'Job description not available')
        
        # Render the HTML template
        return templates.TemplateResponse(
            "job_application.html",
            {
                "request": request,
                "job_id": job_id,
                "job_title": job_title,
                "job_description": job_description
            }
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job post not found: {str(e)}")
    
@app.post("/submit-application/{job_id}")
async def submit_job_application(
    job_id: str,
    background_tasks: BackgroundTasks,
    applicant_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    cover_letter: str = Form(None),
    cv_file: UploadFile = File(...)
):
    """Submit a job application"""
    global faiss_index, metadata
    
    try:
        # Create applications directory if it doesn't exist
        applications_dir = Path("applications")
        applications_dir.mkdir(exist_ok=True)
        
        # Create a directory for this job if it doesn't exist
        job_applications_dir = applications_dir / job_id
        job_applications_dir.mkdir(exist_ok=True)
        
        # Generate a unique ID for this application
        application_id = f"{applicant_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save the CV file
        cv_filename = f"{application_id}.pdf"
        cv_path = job_applications_dir / cv_filename
        
        with open(cv_path, "wb") as buffer:
            buffer.write(await cv_file.read())
        
        # Save application metadata
        application_metadata = {
            "job_id": job_id,
            "applicant_name": applicant_name,
            "email": email,
            "phone": phone,
            "cover_letter": cover_letter,
            "cv_filename": cv_filename,
            "submission_date": datetime.datetime.now().isoformat()
        }
        
        metadata_path = job_applications_dir / f"{application_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(application_metadata, f, indent=2)
        
        # Add the CV to the FAISS index for ranking
        if faiss_index is not None and metadata is not None:
            # Use the full path to the CV file
            cv_full_path = str(cv_path)
            
            # Add the CV to the FAISS index with the original filename
            updated_index, updated_metadata, success, message = add_cv(
                cv_full_path,
                faiss_index,
                metadata,
                cv_filename
            )
            
            if success:
                faiss_index, metadata = updated_index, updated_metadata
                # Update rankings in the background
                background_tasks.add_task(update_rankings)
                print(f"Added application CV {cv_filename} to ranking system")
            else:
                print(f"Failed to add application CV to ranking system: {message}")
        
        return {
            "status": "success",
            "message": "Application submitted successfully",
            "application_id": application_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting application: {str(e)}")

@app.get("/applications/{job_id}")
async def get_job_applications(job_id: str):
    """Get all applications for a specific job"""
    try:
        applications_dir = Path("applications") / job_id
        if not applications_dir.exists():
            return {"applications": []}
        
        applications = []
        for metadata_file in applications_dir.glob("*.json"):
            with open(metadata_file, "r") as f:
                application = json.load(f)
                applications.append(application)
        
        return {"applications": applications}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving applications: {str(e)}")

@app.get("/job-stats/{job_id}")
async def get_job_stats(job_id: str):
    """Get statistics for a specific job post"""
    try:
        # In a real application, you would track views and applications in a database
        # For now, we'll just count the number of applications
        applications_dir = Path("applications") / job_id
        if not applications_dir.exists():
            return {"views": 0, "applications": 0, "conversion_rate": 0}
        
        application_count = len(list(applications_dir.glob("*.json")))
        
        # Mock view count (would be tracked in a real application)
        view_count = application_count * 5  # Assume 5 views per application
        
        conversion_rate = (application_count / view_count * 100) if view_count > 0 else 0
        
        return {
            "views": view_count,
            "applications": application_count,
            "conversion_rate": round(conversion_rate, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job statistics: {str(e)}")
    
# Startup event
@app.on_event("startup")
async def startup_event():
    global faiss_index, metadata, ranked_cvs
    if (faiss_index is None or metadata is None):
        try:
            faiss_index, metadata = initialize_system(cv_dir)
            ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
        except Exception as e:
            print(f"Error initializing system: {str(e)}")

@app.get("/job-requirements/details")
def get_job_details(path: str):
    """Get detailed information about a specific job post"""
    try:
        path = urllib.parse.unquote(path)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Job file not found")
        
        job_text = extract_text_from_pdf(path)
        cleaned_job_text = clean_text(job_text)
        
        # Extract some basic attributes from filename
        filename = os.path.basename(path)
        parts = filename.replace('.pdf', '').split('_')
        
        attributes = {
            "department": "Engineering",  # Default or extract from filename
            "location": "Remote",         # Default or extract from filename
            "employmentType": "Full-time",# Default or extract from filename
            "level": parts[0] if len(parts) > 0 else "Unknown",
            "responsibilities": job_text  # Or parse this from the text
        }
        
        return {
            "filename": filename,
            "requirements": job_text,
            "cleaned_requirements": cleaned_job_text,
            "attributes": attributes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job details: {str(e)}")
