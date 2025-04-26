import os
from .text_processing import extract_text_from_pdf, clean_text, extract_contact_info
from .text_chunking import chunk_text, chunk_cv, extract_sections
import faiss
import numpy as np
import pickle
from config import FAISS_INDEX_PATH, METADATA_PATH, embedding_model, CHUNK_SIZE, CHUNK_OVERLAP
from utils import generate_cv_summary 


# --- Vector DB Management ---
def process_cvs(cv_directory):
    """Process CVs with error handling and chunking"""
    cv_data = []
    if not os.path.exists(cv_directory):
        raise FileNotFoundError(f"CV directory {cv_directory} not found")

    for filename in os.listdir(cv_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(cv_directory, filename)
            raw_text = extract_text_from_pdf(pdf_path)
            if raw_text:
                try:
                    cleaned = clean_text(raw_text)
                    summary = generate_cv_summary(cleaned)  # You can use raw_text or cleaned_text

                    contact = extract_contact_info(raw_text)
                    
                    # Extract sections from the CV
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
                    
                    # Also create a full document embedding for fallback
                    full_embedding = embedding_model.encode([cleaned])[0]
                    
                    cv_data.append({
                        "filename": filename,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned,
                        "embedding": full_embedding,  # Keep the full embedding for backward compatibility
                        "contact": contact,
                        "sections": sections,
                        "chunks": chunks,
                        "chunk_embeddings": chunk_embeddings,
                        "chunk_count": len(chunks),
                        "summary": summary # added by Sheded
                    })
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    return cv_data


def save_data(index, metadata):
    """Safe data serialization"""
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def load_data():
    """Robust data loading"""
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            return faiss.read_index(FAISS_INDEX_PATH), pickle.load(open(METADATA_PATH, 'rb'))
    except Exception as e:
        print(f"Error loading data: {str(e)}")
    return None, None

def initialize_system(cv_directory):
    faiss_index, metadata = load_data()

    if (faiss_index is None):
        cv_data = process_cvs(cv_directory)
        if not cv_data:
            raise ValueError("No valid CVs processed")

        embeddings = np.array([cv["embedding"] for cv in cv_data])
        dimension = embeddings.shape[1]

        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)
        save_data(faiss_index, cv_data)
        metadata = cv_data

    return faiss_index, metadata