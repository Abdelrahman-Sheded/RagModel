import os
import numpy as np
from .text_processing import extract_text_from_pdf, clean_text, extract_contact_info
from .vector_db import save_data
from config import embedding_model

def add_cv(cv_path, faiss_index, metadata, original_filename=None):
    """Add a new CV to the system with chunking support"""
    if not os.path.exists(cv_path) or not cv_path.endswith('.pdf'):
        return faiss_index, metadata, False, f"Invalid CV path: {cv_path}"
        
    try:
        # Extract CV data
        # Use original_filename if provided, otherwise use the basename of cv_path
        filename = original_filename if original_filename else os.path.basename(cv_path)
        raw_text = extract_text_from_pdf(cv_path)
        if not raw_text:
            error_msg = f"Could not extract text from {filename}"
            return faiss_index, metadata, False, error_msg
            
        cleaned = clean_text(raw_text)
        contact = extract_contact_info(raw_text)
        
        # Extract sections from the CV
        from .text_chunking import extract_sections, chunk_text
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        
        sections = extract_sections(raw_text)
        
        # Create chunks from the raw text
        chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Create embeddings for each chunk
        chunk_embeddings = []
        for chunk in chunks:
            chunk_embedding = embedding_model.encode([chunk])[0]
            chunk_embeddings.append({
                "text": chunk,
                "embedding": chunk_embedding
            })
        
        # Also create a full document embedding for backward compatibility
        full_embedding = embedding_model.encode([cleaned])[0]
        
        # Check if CV already exists
        for cv in metadata:
            if cv['filename'] == filename:
                error_msg = f"CV {filename} already exists in the system"
                return faiss_index, metadata, False, error_msg
                
        # Add to metadata
        new_cv = {
            "filename": filename,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "embedding": full_embedding,
            "contact": contact,
            "sections": sections,
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings,
            "chunk_count": len(chunks)
        }
        metadata.append(new_cv)
        
        # Add to FAISS index
        faiss_index.add(np.array([full_embedding]))
        
        # Save updated data
        save_data(faiss_index, metadata)
        return faiss_index, metadata, True, "CV added successfully"
    except Exception as e:
        error_msg = f"Error adding CV: {str(e)}"
        return faiss_index, metadata, False, error_msg

def remove_cv_from_system(filename, faiss_index, metadata):
    """Remove a CV from the system"""
    try:
        # Find CV in metadata
        original_metadata_length = len(metadata)
        found_index = None
        
        for i, cv in enumerate(metadata):
            if cv['filename'] == filename:
                found_index = i
                break
                
        if found_index is None:
            print(f"CV {filename} not found in the system")
            # If we couldn't find the CV, return the original data unchanged
            return faiss_index, metadata
                
        # Remove from metadata
        removed_cv = metadata.pop(found_index)
        print(f"Removed CV {filename} from metadata")
        
        # Rebuild FAISS index (since we can't remove individual vectors)
        if len(metadata) > 0:
            embeddings = np.array([cv["embedding"] for cv in metadata])
            dimension = embeddings.shape[1]
            
            new_index = faiss_index.__class__(dimension)
            new_index.add(embeddings)
            print(f"Rebuilt index with {len(metadata)} vectors")
        else:
            # If there are no CVs left, create an empty index with the same dimension
            dimension = removed_cv["embedding"].shape[0]
            new_index = faiss_index.__class__(dimension)
            print("Created empty index (no CVs remaining)")
            
        # Save updated data
        save_data(new_index, metadata)
        print(f"Saved updated data. Original metadata length: {original_metadata_length}, New length: {len(metadata)}")
        
        return new_index, metadata
    except Exception as e:
        print(f"Error in remove_cv_from_system: {str(e)}")
        # Always return the original index and metadata in case of error
        return faiss_index, metadata