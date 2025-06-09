#!/bin/bash

# Install Python packages
pip install --no-cache-dir -r requirements.txt

# Install spaCy model separately
python -m spacy download en_core_web_sm

# Clean up pip cache
pip cache purge

# Remove unnecessary files
find /usr/local/lib/python3.9/site-packages -name "*.pyc" -delete
find /usr/local/lib/python3.9/site-packages -name "*.pyo" -delete
find /usr/local/lib/python3.9/site-packages -name "*.pyd" -delete
find /usr/local/lib/python3.9/site-packages -name "__pycache__" -delete
find /usr/local/lib/python3.9/site-packages -name "*.dist-info" -delete
find /usr/local/lib/python3.9/site-packages -name "*.egg-info" -delete

# Remove test files
find /usr/local/lib/python3.9/site-packages -name "tests" -type d -exec rm -rf {} +
find /usr/local/lib/python3.9/site-packages -name "test" -type d -exec rm -rf {} +

# Remove documentation
find /usr/local/lib/python3.9/site-packages -name "docs" -type d -exec rm -rf {} +
find /usr/local/lib/python3.9/site-packages -name "*.md" -delete
find /usr/local/lib/python3.9/site-packages -name "*.rst" -delete
find /usr/local/lib/python3.9/site-packages -name "*.txt" -delete

# Remove unnecessary language models from spaCy
find /usr/local/lib/python3.9/site-packages/spacy/lang -type d -not -name "en" -exec rm -rf {} + 