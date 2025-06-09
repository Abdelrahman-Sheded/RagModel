import spacy
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Use Railway's storage path if available, otherwise use local paths
RAILWAY_STORAGE_PATH = os.getenv("RAILWAY_STORAGE_PATH", ".")
FAISS_INDEX_PATH = os.path.join(RAILWAY_STORAGE_PATH, "db", "cv_index.faiss")
METADATA_PATH = os.path.join(RAILWAY_STORAGE_PATH, "db", "cv_metadata.pkl")
CV_DIR = os.path.join(RAILWAY_STORAGE_PATH, "images")

# Create necessary directories
os.makedirs(os.path.join(RAILWAY_STORAGE_PATH, "db"), exist_ok=True)
os.makedirs(CV_DIR, exist_ok=True)

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo-16k")
INITIAL_CANDIDATES = 200
FINAL_RANKING = 50

# Text chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Azure OpenAI Configuration
AZURE_CONFIG = {
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "api_key": os.getenv("AZURE_API_KEY"),
    "api_version": os.getenv("AZURE_API_VERSION")
}