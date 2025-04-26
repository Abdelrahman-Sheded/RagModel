import re
import spacy
from typing import List, Dict, Any

nlp = spacy.load("en_core_web_sm")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of approximately chunk_size characters.
    
    Args:
        text: The text to split into chunks
        chunk_size: The target size of each chunk in characters
        chunk_overlap: The number of characters of overlap between chunks
        
    Returns:
        A list of text chunks
    """
    if not text or chunk_size <= 0:
        return []
        
    # Ensure chunk_overlap is smaller than chunk_size
    chunk_overlap = min(chunk_overlap, chunk_size - 100)
    
    # Split text into sentences using spaCy for better semantic chunking
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence would exceed the chunk size, finalize the current chunk
        if current_size + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep some sentences for overlap with the next chunk
            overlap_size = 0
            overlap_sentences = []
            
            # Work backwards through current_chunk to create overlap
            for i in range(len(current_chunk) - 1, -1, -1):
                sent = current_chunk[i]
                overlap_size += len(sent) + 1  # +1 for space
                overlap_sentences.insert(0, sent)
                
                if overlap_size >= chunk_overlap:
                    break
            
            # Start new chunk with overlap sentences
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        # Add the current sentence to the chunk
        current_chunk.append(sentence)
        current_size += sentence_len + 1  # +1 for space
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def chunk_cv(cv_data: Dict[str, Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    """
    Process a CV dictionary to include chunked text.
    
    Args:
        cv_data: Dictionary containing CV data
        chunk_size: The target size of each chunk
        chunk_overlap: The overlap between chunks
        
    Returns:
        Updated CV dictionary with chunks
    """
    if not cv_data or "raw_text" not in cv_data:
        return cv_data
    
    # Create chunks from the raw text
    chunks = chunk_text(cv_data["raw_text"], chunk_size, chunk_overlap)
    
    # Add chunks to the CV data
    cv_data["chunks"] = chunks
    cv_data["chunk_count"] = len(chunks)
    
    return cv_data

def extract_sections(text: str) -> Dict[str, str]:
    """
    Attempt to extract common CV sections like education, experience, skills, etc.
    
    Args:
        text: The CV text to analyze
        
    Returns:
        Dictionary of section names and their content
    """
    # Common section headers in CVs
    section_patterns = {
        "education": r"(?i)\b(education|academic|qualification|degree)s?\b",
        "experience": r"(?i)\b(experience|employment|work history|professional)\b",
        "skills": r"(?i)\b(skills|technical skills|competencies|expertise)\b",
        "projects": r"(?i)\b(projects|portfolio|works)\b",
        "summary": r"(?i)\b(summary|profile|objective|about me)\b",
        "certifications": r"(?i)\b(certifications|certificates|accreditations)\b",
        "languages": r"(?i)\b(languages|language proficiency)\b",
        "contact": r"(?i)\b(contact|personal details|personal information)\b"
    }
    
    sections = {}
    
    # Try to find each section in the text
    for section_name, pattern in section_patterns.items():
        matches = re.finditer(pattern, text)
        
        for match in matches:
            start_pos = match.start()
            
            # Find the next section header after this one
            next_section_pos = len(text)
            for other_pattern in section_patterns.values():
                other_matches = re.finditer(other_pattern, text[start_pos + 1:])
                for other_match in other_matches:
                    next_pos = start_pos + 1 + other_match.start()
                    if next_pos < next_section_pos:
                        next_section_pos = next_pos
            
            # Extract the section content
            section_content = text[start_pos:next_section_pos].strip()
            
            # Store or append to the section
            if section_name in sections:
                sections[section_name] += "\n" + section_content
            else:
                sections[section_name] = section_content
    
    return sections