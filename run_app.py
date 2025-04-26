import subprocess
import os
import time
import sys
import webbrowser
from threading import Thread
import requests
import socket

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_api_ready(max_wait=30):
    """Wait for FastAPI to be ready"""
    print("Waiting for FastAPI server to be ready...")
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("FastAPI server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print(f"Warning: FastAPI server not responding after {max_wait} seconds")
    return False

def run_api():
    print("Starting FastAPI server...")
    subprocess.run(["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"], check=True)

def run_streamlit():
    print("Starting Streamlit app...")
    # Add the --server.headless=true flag to prevent Streamlit from opening a browser automatically
    subprocess.run(["streamlit", "run", "app/streamlit_app.py", "--server.headless=true"], check=True)

def open_browser():
    print("Opening Streamlit in browser...")
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    print("Starting CV Chatbot application...")
    
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Install requirements if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Check if the API port is already in use
    if is_port_in_use(8000):
        print("Warning: Port 8000 is already in use. API server may already be running.")
    else:
        # Start API in a separate thread
        api_thread = Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        
        # Wait for API to be ready (with a timeout)
        wait_for_api_ready(max_wait=30)
    
    # Open browser after API is ready
    browser_thread = Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Streamlit (in main thread)
    run_streamlit()
