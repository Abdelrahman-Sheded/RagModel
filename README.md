# CV Ranking System

A smart AI-powered application that helps recruiters match job candidates to positions by analyzing and ranking CVs based on job requirements.

## Overview

This CV Ranking System uses natural language processing and vector similarity search to automatically analyze resumes, extract key information, and rank candidates against specific job requirements. The system provides an intuitive interface for recruiters to manage the hiring process efficiently.

## Key Features

- **CV Analysis & Ranking**: Automatically analyzes CVs and ranks candidates based on similarity to job requirements
- **Interactive Chat**: AI assistant to answer questions about candidates and provide insights
- **Candidate Comparison**: Side-by-side comparison of candidates with detailed analysis
- **Job Post Management**: Create and manage job posts with customizable requirements
- **Application Portal**: Shareable links for candidates to submit applications directly
- **Contact Information Extraction**: Automatically extracts email and phone from CVs

## Technology Stack

- **Backend**: FastAPI for RESTful API services
- **Frontend**: Streamlit for interactive web interface
- **Vector Search**: FAISS for efficient similarity search
- **NLP**: Text processing and embedding models for CV analysis
- **AI Chat**: Integration with Azure OpenAI for intelligent candidate analysis
- **Data Storage**: File-based storage for CVs, applications, and job descriptions

## Project Structure

The project has been reorganized into a more maintainable structure:

```
CV_ranking/
├── api/            # API-related code (FastAPI endpoints)
├── app/            # Frontend application (Streamlit)
├── db/             # Database files (FAISS index, metadata)
├── src/            # Core source code
│   ├── chat.py     # Chat functionality
│   ├── ranking.py  # CV ranking algorithms
│   ├── text_processing.py  # Text extraction and processing
│   └── vector_db.py  # Vector database operations
├── utils/          # Utility functions and helpers
├── config.py       # Configuration settings
├── requirements.txt  # Project dependencies
├── .env            # Environment variables (not in version control)
└── README.md       # Project documentation
```

## Getting Started

To run the application:

1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file with your Azure OpenAI credentials:
   ```
   AZURE_ENDPOINT=your_endpoint
   AZURE_API_KEY=your_api_key
   AZURE_API_VERSION=2024-08-01-preview
   DEPLOYMENT_NAME=gpt-35-turbo-16k
   ```
3. Run the application: `python run_app.py`

The system will automatically index existing CVs in the `images` directory and use the job description specified in the configuration.