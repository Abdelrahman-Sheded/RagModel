import os
import shutil
from pathlib import Path
import subprocess
import sys

def upload_to_railway():
    """
    Upload data to Railway's persistent storage
    """
    try:
        # Get the current directory
        current_dir = Path.cwd()
        source_dir = current_dir / "images"
        railway_data_dir = Path("/")

        # Check if source directory exists
        if not source_dir.exists():
            print(f"Error: Source directory {source_dir} does not exist")
            return False

        # Create destination directory if it doesn't exist
        os.makedirs(railway_data_dir / "images", exist_ok=True)
        os.makedirs(railway_data_dir / "db", exist_ok=True)

        # Copy files to Railway storage
        print("Copying files to Railway storage...")
        for file in source_dir.glob("*"):
            if file.is_file():
                dest_path = railway_data_dir / "images" / file.name
                shutil.copy2(file, dest_path)
                print(f"Copied {file.name} to Railway storage")

        print("Data upload completed successfully!")
        return True

    except Exception as e:
        print(f"Error uploading data: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if running in Railway environment
    if not os.path.exists("/images"):
        print("Not running in Railway environment. Please run this script after deploying to Railway.")
        sys.exit(1)

    success = upload_to_railway()
    if not success:
        sys.exit(1) 