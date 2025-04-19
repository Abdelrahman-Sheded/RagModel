import re
import faiss
import numpy as np
from .text_processing import extract_text_from_pdf, clean_text
from config import embedding_model, INITIAL_CANDIDATES, FINAL_RANKING, AZURE_CONFIG, DEPLOYMENT_NAME
from langchain_openai import AzureChatOpenAI

def truncate_text(text, max_length=1000):
    return text[:max_length] + '...' if len(text) > max_length else text

def parse_llm_response(response, max_candidates):
    try:
        numbers = [int(num.strip()) for num in re.findall(r'\b\d+\b', response)]
        valid_numbers = [n for n in numbers if 1 <= n <= max_candidates]
        return [n-1 for n in valid_numbers][:max_candidates]
    except:
        return []

def rank_cvs(job_description_path, faiss_index, metadata, top_n=50):
    raw_jd = extract_text_from_pdf(job_description_path)
    cleaned_jd = clean_text(raw_jd)
    if not cleaned_jd:
        raise ValueError("Invalid job description")

    # First, get initial candidates using the full document embedding for efficiency
    jd_embedding = embedding_model.encode([cleaned_jd])[0]
    distances, indices = faiss_index.search(np.array([jd_embedding]), INITIAL_CANDIDATES)

    # Create initial candidate list with basic similarity scores
    initial_candidates = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            initial_candidates.append({
                **metadata[idx],
                "similarity": 1 / (1 + distances[0][i])
            })
    
    if not initial_candidates:
        return []
    
    # Now perform a more detailed analysis using chunks and LLM
    # Prepare a more detailed prompt with relevant chunks from each candidate
    detailed_candidate_info = []
    
    # Take top candidates from initial screening for detailed analysis
    top_initial_candidates = initial_candidates[:min(20, len(initial_candidates))]
    
    for i, cv in enumerate(top_initial_candidates):
        # Extract the most relevant sections/chunks from the CV
        relevant_sections = ""
        
        # Include education section if available
        if "sections" in cv and "education" in cv["sections"]:
            relevant_sections += f"Education:\n{truncate_text(cv['sections']['education'], 500)}\n\n"
        
        # Include experience section if available
        if "sections" in cv and "experience" in cv["sections"]:
            relevant_sections += f"Experience:\n{truncate_text(cv['sections']['experience'], 1000)}\n\n"
        
        # Include skills section if available
        if "sections" in cv and "skills" in cv["sections"]:
            relevant_sections += f"Skills:\n{truncate_text(cv['sections']['skills'], 500)}\n\n"
        
        # If no sections were found, use the most relevant chunks
        if not relevant_sections and "chunks" in cv and cv["chunks"]:
            # Use the first 2-3 chunks as a fallback
            for j, chunk in enumerate(cv["chunks"][:3]):
                relevant_sections += f"Chunk {j+1}:\n{truncate_text(chunk, 500)}\n\n"
        
        # If still no relevant content, use the cleaned text
        if not relevant_sections:
            relevant_sections = truncate_text(cv["cleaned_text"], 2000)
        
        candidate_info = f"[Candidate {i+1}]\nFile: {cv['filename']}\n"
        if "contact" in cv and cv["contact"]:
            candidate_info += f"Contact: {cv['contact'].get('email', 'N/A')} | {cv['contact'].get('phone', 'N/A')}\n"
        candidate_info += f"\nProfile:\n{relevant_sections}"
        
        detailed_candidate_info.append(candidate_info)
    
    # Create a detailed prompt for the LLM to analyze candidates
    # Fix the backslash issue by preparing the joined string separately
    candidate_profiles = "\n\n".join(detailed_candidate_info)
    
    detailed_prompt = f"""You are an expert recruiter tasked with finding the best candidates for a job position.

Job Requirements:
{raw_jd[:2000]}

Candidate Profiles:
{candidate_profiles}

Analyze each candidate's qualifications, experience, and skills in relation to the job requirements.
Consider factors such as relevant experience, technical skills, education, and overall fit for the position.

Rank the top {FINAL_RANKING} most suitable candidates by their numbers (1-{len(detailed_candidate_info)}).
Provide your ranking as a comma-separated list of candidate numbers in order of suitability (best first).
Only output the numbers, separated by commas."""

    # Use the LLM to rank candidates based on detailed analysis
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_CONFIG["azure_endpoint"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        deployment_name=DEPLOYMENT_NAME,
        temperature=0
    )

    response = llm.invoke(detailed_prompt)
    selected_indices = parse_llm_response(response, len(top_initial_candidates))
    
    # Return the ranked candidates
    if selected_indices:
        return [top_initial_candidates[i] for i in selected_indices]
    else:
        # Fallback to initial ranking if LLM doesn't provide valid indices
        return top_initial_candidates[:FINAL_RANKING]