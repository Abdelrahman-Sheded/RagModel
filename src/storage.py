from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
from pathlib import Path
import tempfile
import shutil

class AzureStorage:
    def __init__(self, connection_string):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = "rag-model"
        self._ensure_container()

    def _ensure_container(self):
        """Ensure the container exists"""
        try:
            self.blob_service_client.create_container(self.container_name)
        except Exception:
            # Container might already exist
            pass

    def upload_file(self, local_path, blob_name):
        """Upload a file to Azure Blob Storage"""
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        return blob_client.url

    def download_file(self, blob_name, local_path):
        """Download a file from Azure Blob Storage"""
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        with open(local_path, "wb") as file:
            file.write(blob_client.download_blob().readall())

    def download_directory(self, prefix, local_dir):
        """Download all files with a specific prefix to a local directory"""
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # List all blobs with the prefix
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        
        for blob in blob_list:
            # Get the relative path from the prefix
            relative_path = blob.name[len(prefix):].lstrip('/')
            if not relative_path:
                continue
                
            # Create the full local path
            local_path = os.path.join(local_dir, relative_path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the blob
            blob_client = container_client.get_blob_client(blob.name)
            with open(local_path, "wb") as file:
                file.write(blob_client.download_blob().readall())

    def upload_directory(self, local_dir, prefix):
        """Upload all files from a local directory to Azure Blob Storage"""
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Get the relative path from the local directory
                relative_path = os.path.relpath(local_path, local_dir)
                # Create the blob name with the prefix
                blob_name = f"{prefix}/{relative_path}".replace("\\", "/")
                
                # Upload the file
                with open(local_path, "rb") as data:
                    container_client.upload_blob(
                        name=blob_name,
                        data=data,
                        overwrite=True
                    )

def initialize_storage():
    """Initialize Azure Storage with environment variables"""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
    
    return AzureStorage(connection_string)

def download_model_files():
    """Download model files from Azure Storage"""
    storage = initialize_storage()
    
    # Create temporary directory for model files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download model files
        storage.download_directory("model", temp_dir)
        
        # Move files to their final locations
        model_dir = Path("db")
        model_dir.mkdir(exist_ok=True)
        
        # Move files from temp directory to final location
        for file in Path(temp_dir).glob("**/*"):
            if file.is_file():
                dest_path = model_dir / file.relative_to(temp_dir)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(dest_path))

def upload_model_files():
    """Upload model files to Azure Storage"""
    storage = initialize_storage()
    
    # Upload model files
    storage.upload_directory("db", "model") 