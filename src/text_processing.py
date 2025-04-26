import re
import PyPDF2
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files with error handling"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Error reading {pdf_path}: {str(e)}")
        return ""

def clean_text(text):
    """Clean text using spaCy with validation"""
    if not text:
        return ""
    try:
        doc = nlp(text)
        return ' '.join([token.lemma_.lower() for token in doc
                        if not token.is_stop and not token.is_punct and not token.is_space])
    except Exception as e:
        print(f"Text cleaning error: {str(e)}")
        return ""

def extract_contact_info(text):
    """Extract contact info with validation"""
    try:
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        phones = re.findall(r'\+?\d[\d -]{7,}\d', text)
        return {
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None
        }
    except Exception as e:
        print(f"Contact extraction error: {str(e)}")
        return {'email': None, 'phone': None}