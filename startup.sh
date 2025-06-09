#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 