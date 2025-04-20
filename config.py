import spacy
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Update paths to use db directory
FAISS_INDEX_PATH = os.path.join("db", "cv_index.faiss")
METADATA_PATH = os.path.join("db", "cv_metadata.pkl")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo-16k")

INITIAL_CANDIDATES = 150  # Reduced from 150
FINAL_RANKING = 20
CHUNK_SIZE = 1000  # Reduced from 1000
CHUNK_OVERLAP = 200  # Reduced from 200

# Azure OpenAI Configuration
AZURE_CONFIG = {
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "api_key": os.getenv("AZURE_API_KEY"),
    "api_version": os.getenv("AZURE_API_VERSION")
}