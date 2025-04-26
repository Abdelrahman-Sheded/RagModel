# main.py
import os
import sys

# Add the project root directory to Python's module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_db import initialize_system
from src.chat import interactive_chat

def main():
    cv_dir = "images"
    job_desc_path = "junior_devops_requirements.pdf"
    
    try:
        faiss_index, metadata = initialize_system(cv_dir)
        interactive_chat(faiss_index, metadata, job_desc_path)
    except Exception as e:
        print(f"System initialization failed: {str(e)}")

if __name__ == "__main__":
    main()