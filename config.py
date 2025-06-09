import spacy
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import tempfile
import shutil
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "rag-model"

# Local paths
DB_DIR = "db"
FAISS_INDEX_PATH = os.path.join(DB_DIR, "cv_index.faiss")
METADATA_PATH = os.path.join(DB_DIR, "cv_metadata.pkl")

# Azure paths
AZURE_FAISS_INDEX_PATH = "model/cv_index.faiss"
AZURE_METADATA_PATH = "model/cv_metadata.pkl"

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo-16k")

INITIAL_CANDIDATES = 150
FINAL_RANKING = 20
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Azure OpenAI Configuration
AZURE_CONFIG = {
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "api_key": os.getenv("AZURE_API_KEY"),
    "api_version": os.getenv("AZURE_API_VERSION")
}

def ensure_model_files():
    """Ensure model files are available locally, downloading from Azure if needed"""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    
    # Check if files exist locally
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        print("Model files not found locally, downloading from Azure Storage...")
        
        if not AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
        
        # Initialize Azure Storage client
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Download files
        for azure_path, local_path in [
            (AZURE_FAISS_INDEX_PATH, FAISS_INDEX_PATH),
            (AZURE_METADATA_PATH, METADATA_PATH)
        ]:
            try:
                blob_client = container_client.get_blob_client(azure_path)
                with open(local_path, "wb") as file:
                    file.write(blob_client.download_blob().readall())
                print(f"Downloaded {azure_path} to {local_path}")
            except Exception as e:
                print(f"Error downloading {azure_path}: {str(e)}")
                raise