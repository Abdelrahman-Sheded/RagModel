import os
import sys
from pathlib import Path

# Add the project root directory to Python's module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dotenv exactly like config.py
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContainerClient

# Load environment variables at module level, just like config.py
load_dotenv()

def initialize_storage():
    """Initialize Azure Storage with environment variables"""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
    
    return BlobServiceClient.from_connection_string(connection_string)

def upload_model_files():
    """Upload model files to Azure Storage"""
    # Initialize storage
    blob_service_client = initialize_storage()
    container_name = "rag-model"
    
    # Ensure container exists
    try:
        blob_service_client.create_container(container_name)
    except Exception:
        # Container might already exist
        pass
    
    container_client = blob_service_client.get_container_client(container_name)
    
    # Upload model files
    for root, _, files in os.walk("db"):
        for file in files:
            local_path = os.path.join(root, file)
            # Get the relative path from the db directory
            relative_path = os.path.relpath(local_path, "db")
            # Create the blob name with the model prefix
            blob_name = f"model/{relative_path}".replace("\\", "/")
            
            print(f"Uploading {local_path} to {blob_name}")
            # Upload the file
            with open(local_path, "rb") as data:
                container_client.upload_blob(
                    name=blob_name,
                    data=data,
                    overwrite=True
                )

def main():
    if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        print("Please set AZURE_STORAGE_CONNECTION_STRING in your .env file")
        return
    
    print("Uploading model files to Azure Storage...")
    try:
        upload_model_files()
        print("Upload completed successfully!")
    except Exception as e:
        print(f"Error uploading files: {str(e)}")

if __name__ == "__main__":
    # Make sure we're in the right directory, just like run_app.py does
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()