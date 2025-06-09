#!/bin/bash

# Print current directory and contents for debugging
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

# Create and activate virtual environment
echo "Setting up virtual environment..."
python -m venv antenv
source antenv/bin/activate

# Install Python dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Start the FastAPI application with gunicorn
echo "Starting FastAPI application..."
gunicorn api.test_api:app --bind 0.0.0.0:8000 --workers 4 --timeout 120 --log-level debug 