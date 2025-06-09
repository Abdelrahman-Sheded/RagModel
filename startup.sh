#!/bin/bash

# Print current directory and contents for debugging
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Start the FastAPI application
echo "Starting FastAPI application..."
uvicorn api.test_api:app --host 0.0.0.0 --port 8000 --log-level debug 