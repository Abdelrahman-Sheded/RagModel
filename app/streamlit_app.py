import streamlit as st
import requests
import pandas as pd
import json
import os
import re
import time
import backoff

# Configuration
API_URL = "http://localhost:8000"  # Change to match your FastAPI URL

# App title and description
st.set_page_config(
    page_title="CV Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CV Chatbot")
st.markdown("Upload CVs, rank candidates, and chat with the AI assistant.")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "candidates" not in st.session_state:
    st.session_state.candidates = []

if "current_candidate_id" not in st.session_state:
    st.session_state.current_candidate_id = None

if "api_ready" not in st.session_state:
    st.session_state.api_ready = False

# Add retry mechanism for API connections
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5, max_time=20)
def api_request(method, endpoint, **kwargs):
    """Make API request with retry logic"""
    url = f"{API_URL}/{endpoint}"
    response = method(url, **kwargs)
    response.raise_for_status()
    return response

# Function to check API health with retries
def check_api_health():
    try:
        health_response = api_request(requests.get, "health")
        if health_response.status_code == 200:
            st.session_state.api_ready = True
            return True, health_response.json()
        return False, None
    except Exception as e:
        return False, str(e)

# Function to load candidates from the API
def load_candidates(top_n=20):
    try:
        if not st.session_state.api_ready:
            is_healthy, _ = check_api_health()
            if not is_healthy:
                return []
        
        response = api_request(requests.get, f"candidates?top_n={top_n}")
        if response.status_code == 200:
            st.session_state.candidates = response.json()["candidates"]
            return st.session_state.candidates
        else:
            st.error(f"Error loading candidates: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

# Function to send a chat message to the API with streaming response
def send_chat_message(messages, response_placeholder=None):
    try:
        if response_placeholder:
            # Simulate streaming response if placeholder is provided
            response_placeholder.markdown("▌")
            
            response = requests.post(
                f"{API_URL}/chat",
                json={"messages": messages}
            )
            
            if response.status_code == 200:
                full_response = response.json()["response"]
                # Stream the response word by word to simulate typing
                displayed_response = ""
                for word in full_response.split(" "):
                    displayed_response += word + " "
                    response_placeholder.markdown(displayed_response + "▌")
                    time.sleep(0.01)  # Small delay between words
                
                # Final response without cursor
                response_placeholder.markdown(full_response)
                return full_response
            else:
                response_placeholder.markdown(f"Error from chat API: {response.text}")
                return "I'm sorry, I encountered an error processing your request."
        else:
            # Non-streaming mode (for Quick Actions)
            response = requests.post(
                f"{API_URL}/chat",
                json={"messages": messages}
            )
            if response.status_code == 200:
                return response.json()["response"]
            else:
                st.error(f"Error from chat API: {response.text}")
                return "I'm sorry, I encountered an error processing your request."
    except Exception as e:
        if response_placeholder:
            response_placeholder.markdown(f"Error connecting to chat API: {str(e)}")
        else:
            st.error(f"Error connecting to chat API: {str(e)}")
        return "I'm sorry, I couldn't connect to the chat service."

# Function to get job requirements from the API
def get_job_requirements():
    try:
        response = requests.get(f"{API_URL}/job-requirements")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error loading job requirements: {response.text}")
            return {"requirements": "Unable to fetch job requirements"}
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return {"requirements": "Unable to connect to job requirements API"}

# Add function to get list of job requirements
def get_job_requirements_list():
    try:
        response = requests.get(f"{API_URL}/job-requirements/list")
        if response.status_code == 200:
            return response.json()["job_files"]
        else:
            st.error(f"Error loading job requirements list: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

# Function to build a system prompt with job requirements and candidate information
def build_system_prompt():
    try:
        # Fetch job requirements
        job_info = get_job_requirements()
        job_requirements = job_info.get("requirements", "Job requirements not available")
        
        # Fetch the top candidates from the API
        response = requests.get(f"{API_URL}/candidates?top_n=10")
        if response.status_code == 200:
            candidates = response.json()["candidates"]
            
            # Create a system prompt that includes job requirements and candidate information
            system_prompt = "You're an AI assistant that helps analyze and compare candidates for a specific job. Here is information about the job and top candidates:\n\n"
            
            # Add job requirements section
            system_prompt += "JOB REQUIREMENTS:\n"
            system_prompt += f"{job_requirements}\n\n"
            system_prompt += "====================\n\n"
            
            # Add candidates section with clear ranking explanation
            system_prompt += "TOP CANDIDATES (ranked from best to worst match):\n"
            system_prompt += "Candidates are numbered by their rank, where Candidate 1 is the best match for the job requirements, Candidate 2 is the second best, and so on.\n"
            system_prompt += "Lower numbered candidates should generally be considered better matches than higher numbered candidates.\n\n"
            
            for i, candidate in enumerate(candidates):
                rank = i + 1
                system_prompt += f"Candidate {rank} (Rank #{rank}): {candidate['filename']}\n"
                system_prompt += f"Similarity Score: {candidate['similarity']:.2f}\n"
                system_prompt += f"Contact: Email: {candidate['contact']['email'] or 'N/A'}, Phone: {candidate['contact']['phone'] or 'N/A'}\n"
                system_prompt += f"Summary: {candidate['summary']}\n\n"
            
            system_prompt += "When comparing candidates, always remember that lower-ranked candidates (with smaller numbers) have been determined to be better matches for the job requirements than higher-ranked candidates (with larger numbers), unless new information in the conversation overrides this ranking.\n\n"
            
            # Add instruction to avoid "No information provided" responses
            system_prompt += "RESPONSE GUIDELINES:\n"
            system_prompt += "- Never respond with phrases like 'No information provided' or 'No information available'\n"
            system_prompt += "- If specific information isn't explicitly mentioned in a candidate's profile, make reasonable inferences based on their background, skills, and experience\n"
            system_prompt += "- If you're genuinely uncertain about a detail, say something like 'While not explicitly mentioned in their CV, based on their background in [relevant field], they likely...' or 'Their CV doesn't highlight this specific aspect, but...'\n"
            system_prompt += "- Always provide a helpful response that offers insights based on available information\n"
            system_prompt += "- If asked about very specific details not in the CV, acknowledge the limitation briefly and provide related information that IS available\n\n"
            
            # Add specific guidelines for comparing candidates with improved formatting
            system_prompt += "COMPARISON FORMAT GUIDELINES:\n"
            system_prompt += "- When comparing candidates, use a conversational paragraph style instead of formal sections\n" 
            system_prompt += "- DO NOT use a rigid structure with headings like 'Education:', 'Technical Skills:', 'Soft Skills:'\n"
            system_prompt += "- Instead, write a flowing analysis that naturally incorporates relevant information from both candidates\n"
            system_prompt += "- Begin with a brief introduction of both candidates and their overall fit for the position\n"
            system_prompt += "- Follow with direct comparisons of their relevant qualifications, focusing on job requirements\n"
            system_prompt += "- Conclude with your assessment of which candidate appears better suited and why\n"
            system_prompt += "- Focus only on information that is available and relevant to the job requirements\n"
            system_prompt += "- Never mention the absence of information - simply discuss what IS present in their profiles\n"
            
            return system_prompt
        else:
            return "You're an AI assistant that helps analyze candidates. However, I couldn't fetch candidate information at this time."
    except Exception as e:
        return "You're an AI assistant that helps analyze candidates. However, I couldn't fetch candidate information at this time."

# Check API health with retry logic
api_status_container = st.sidebar.empty()

if not st.session_state.api_ready:
    with api_status_container:
        st.warning("⏳ Connecting to API...")
    is_ready, health_data = check_api_health()
    if is_ready:
        with api_status_container:
            st.success("✅ Connected to API")
            cv_count = health_data.get("cv_count", 0)
            st.info(f"📁 {cv_count} CVs in database")
    else:
        with api_status_container:
            st.error("❌ Cannot connect to API")
            st.info("The API may still be starting up. Try refreshing in a moment.")
else:
    with api_status_container:
        st.success("✅ Connected to API")
        try:
            health_response = api_request(requests.get, "health")
            cv_count = health_response.json().get("cv_count", 0)
            st.info(f"📁 {cv_count} CVs in database")
        except:
            st.warning("Cannot retrieve current CV count")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["CV Rankings", "Candidate Detail", "Chat with AI", "Job Post", "Upload CV"])

# CV Rankings Page
if page == "CV Rankings":
    st.header("Top Candidate Rankings")
    
    # Change slider range from 5-50 to 3-20
    top_n = st.slider("Number of candidates to display:", min_value=3, max_value=20, value=10)
    if st.button("Refresh Rankings"):
        candidates = load_candidates(top_n)
    else:
        candidates = st.session_state.candidates or load_candidates(top_n)
    
    if candidates:
        # Convert to DataFrame for better display - Make ID start from 1 instead of 0
        df = pd.DataFrame([
            {
                "ID": i + 1,  # Add 1 to ID to start from 1 instead of 0
                "Filename": c["filename"],
                "Similarity Score": f"{c['similarity']:.2f}",
                "Email": c["contact"]["email"] or "N/A",
                "Phone": c["contact"]["phone"] or "N/A"
            } for i, c in enumerate(candidates)
        ])
        
        st.dataframe(df, use_container_width=True)
        
        # Delete CV feature
        st.subheader("Remove CV")
        cv_to_delete = st.selectbox("Select CV to remove:", 
                                    [c["filename"] for c in candidates],
                                    key="delete_cv_selector")
        
        if st.button("Remove Selected CV", key="remove_cv_button"):
            try:
                with st.spinner(f"Removing {cv_to_delete}..."):
                    response = requests.delete(f"{API_URL}/remove-cv/{cv_to_delete}")
                    
                    if response.status_code == 200:
                        st.success(f"Successfully removed {cv_to_delete}")
                        # Refresh the candidates list
                        st.session_state.candidates = load_candidates(top_n)
                        # Force UI refresh
                        st.rerun()
                    elif response.status_code == 404:
                        error_detail = response.json().get("detail", "CV not found")
                        st.warning(f"⚠️ {error_detail}")
                    else:
                        try:
                            error_detail = response.json().get("detail", "Unknown error")
                        except:
                            error_detail = f"Status code: {response.status_code}"
                        st.error(f"❌ Error: {error_detail}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
    else:
        st.info("No candidates available. Please upload CVs first.")

# Candidate Detail Page
elif page == "Candidate Detail":
    st.header("Candidate Details")
    
    if not st.session_state.candidates:
        candidates = load_candidates()
    else:
        candidates = st.session_state.candidates
    
    if candidates:
        # Change display indices to start from 1 instead of 0
        candidate_options = {f"{i+1}. {c['filename']}": i for i, c in enumerate(candidates)}
        selected_candidate = st.selectbox(
            "Select a candidate:", 
            options=list(candidate_options.keys()),
            key="candidate_detail_selector"
        )
        
        candidate_id = candidate_options[selected_candidate]
        
        # Get detailed information about the selected candidate
        try:
            response = requests.get(f"{API_URL}/candidates/{candidate_id}")
            if response.status_code == 200:
                candidate_data = response.json()
                
                # Display candidate information
                st.subheader(f"Candidate: {candidate_data['filename']}")
                st.markdown(f"**Similarity Score:** {candidate_data['similarity']:.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Email:** {candidate_data['contact']['email'] or 'N/A'}")
                with col2:
                    st.markdown(f"**Phone:** {candidate_data['contact']['phone'] or 'N/A'}")
                
                st.subheader("CV Content")
                with st.expander("Show Full CV Text", expanded=False):
                    st.text_area(
                        label="CV Full Text",
                        value=candidate_data['full_text'], 
                        height=400,
                        label_visibility="collapsed"  # Hide the label visually
                    )
                
                # Removed the "Cleaned Text (for Similarity Matching)" section
                
            else:
                st.error(f"Error fetching candidate details: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
    else:
        st.info("No candidates available. Please upload CVs first.")

# Chat with AI Page
elif page == "Chat with AI":
    st.header("Chat with AI About Candidates")
    
    # Initialize or reset system message with candidate information
    if not st.session_state.messages or st.button("Refresh Context"):
        system_prompt = build_system_prompt()
        # Reset messages with the new system message
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.success("AI context updated with current job requirements and candidate information")
    
    # Display information about the context being used
    with st.expander("View AI Context", expanded=False):
        if st.session_state.messages and len(st.session_state.messages) > 0:
            st.text_area(
                label="System Context",
                value=st.session_state.messages[0]["content"], 
                height=300,
                label_visibility="collapsed"  # Hide the label visually
            )
    
    # Add tabs for chat and job requirements AFTER the chat input
    tab1, tab2 = st.tabs(["Chat History", "Job Requirements"])
    
    with tab1:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] != "system":  # Don't show system messages
                with st.chat_message(message["role"]):
                    st.write(message["content"])
    
    with tab2:
        job_info = get_job_requirements()
        st.subheader(f"Job Requirements: {job_info.get('filename', 'Unknown')}")
        st.text_area(
            label="Job Requirements Text",
            value=job_info.get("requirements", "Unable to load job requirements"), 
            height=300,
            label_visibility="collapsed"  # Hide the label visually
        )
    
    # User input - this must be at the main level, not within tabs, columns, expanders, etc.
    if prompt := st.chat_input("Ask about candidates..."):
        # Add user message to chat history
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Display user message
        st.chat_message("user").write(prompt)
        
        # Get AI response with streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming response
            response_placeholder = st.empty()
            response = send_chat_message(st.session_state.messages, response_placeholder)
        
        # Add assistant message to chat history
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)
    
    # Quick actions (with modified streaming for better user experience)
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Who's the best candidate?"):
            prompt = "Who is the best candidate for the job and why? Consider the job requirements and explain how their qualifications match."
            # Add user message to chat history
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Get AI response with streaming
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = send_chat_message(st.session_state.messages, response_placeholder)
            
            # Add assistant message to chat history  
            assistant_message = {"role": "assistant", "content": response}
            st.session_state.messages.append(assistant_message)
    
    with col2:
        if st.button("Top 3 candidates?"):
            prompt = "List the top 3 candidates and their key strengths in relation to the job requirements."
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Get AI response with streaming
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = send_chat_message(st.session_state.messages, response_placeholder)
            
            assistant_message = {"role": "assistant", "content": response}
            st.session_state.messages.append(assistant_message)
            
    with col3:
        if st.button("Compare Top 2"):
            prompt = "Compare the top 2 candidates in terms of how well they match the job requirements and tell me which one is better for the job and why."
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Get AI response with streaming
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = send_chat_message(st.session_state.messages, response_placeholder)
            
            assistant_message = {"role": "assistant", "content": response}
            st.session_state.messages.append(assistant_message)
    
    with col4:
        if st.button("Clear Chat"):
            # Reset to just the system message with refreshed information
            system_prompt = build_system_prompt()
            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            st.rerun()

# Job Post Page
elif page == "Job Post":
    st.header("Job Post Management")
    
    # Create tabs for different job post functionality
    job_tabs = st.tabs(["Current Job Post", "Upload New Job Post", "Create from Text", "Manage Job Posts", "Share Job Post"])
    
    # Tab 1: Current Job Post
    with job_tabs[0]:
        job_info = get_job_requirements()
        
        st.subheader(f"Current Job Post: {job_info.get('filename', 'Unknown')}")
        
        with st.expander("View Full Job Description", expanded=True):
            st.text_area(
                label="Job Post Text",
                value=job_info.get("requirements", "Unable to load job post"), 
                height=300,
                label_visibility="collapsed"  # Hide the label visually
            )
        
        # The sections for Education & Experience and Required Skills have been removed as requested
    
    # Tab 2: Upload New Job Post (PDF)
    with job_tabs[1]:
        st.subheader("Upload New Job Post PDF")
        
        with st.form("upload_job_post_form"):
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
            uploaded_file = st.file_uploader("Upload Job Post PDF", type="pdf")
            
            submitted = st.form_submit_button("Upload Job Post")
            if submitted and uploaded_file and job_title:
                try:
                    files = {"file": uploaded_file}
                    response = requests.post(
                        f"{API_URL}/job-requirements/upload-pdf",
                        files=files,
                        data={"title": job_title}
                    )
                    
                    if response.status_code in [200, 201]:
                        st.success(f"Job post uploaded successfully! The system has been updated to use the new job post.")
                        # Force a refresh of the system context
                        if "messages" in st.session_state:
                            st.session_state.messages = []
                    else:
                        st.error(f"Error uploading job post: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
            elif submitted:
                st.warning("Please provide both a job title and a PDF file")
    
    # Tab 3: Create from Text
    with job_tabs[2]:
        st.subheader("Create Job Post from Text")
        
        with st.form("create_job_post_form"):
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
            job_description = st.text_area(
                "Job Description", 
                height=300,
                placeholder="Enter the full job description here..."
            )
            
            submitted = st.form_submit_button("Create Job Post")
            if submitted and job_title and job_description:
                try:
                    response = requests.post(
                        f"{API_URL}/job-requirements/update-text",
                        json={
                            "title": job_title,
                            "requirements_text": job_description
                        }
                    )
                    
                    if response.status_code in [200, 201]:
                        st.success(f"Job post created successfully! The system has been updated to use the new job post.")
                        # Force a refresh of the system context
                        if "messages" in st.session_state:
                            st.session_state.messages = []
                    else:
                        st.error(f"Error creating job post: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
            elif submitted:
                st.warning("Please provide both a job title and job description")
    
    # Tab 4: Manage Job Posts
    with job_tabs[3]:
        st.subheader("Manage Available Job Posts")
        
        job_files = get_job_requirements_list()
        
        if not job_files:
            st.info("No job post files found. Upload or create new job posts to get started.")
        else:
            st.write(f"Found {len(job_files)} job post file(s):")
            
            for job in job_files:
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        status = "✅ ACTIVE" if job["is_current"] else ""
                        st.write(f"**{job['filename']}** {status}")
                    
                    with col2:
                        st.write(f"Created: {job['created']}")
                    
                    with col3:
                        if not job["is_current"]:
                            if st.button("Set Active", key=f"set_active_{job['path']}"):
                                try:
                                    # URL encode the path to handle special characters
                                    import urllib.parse
                                    encoded_path = urllib.parse.quote(job['path'])
                                    
                                    response = requests.post(f"{API_URL}/job-requirements/set-active/{encoded_path}")
                                    if response.status_code == 200:
                                        st.success(f"Successfully set {job['filename']} as the active job post!")
                                        # Force a refresh of the system context
                                        if "messages" in st.session_state:
                                            st.session_state.messages = []
                                        # Refresh the page
                                        st.rerun()
                                    else:
                                        st.error(f"Error setting active job post: {response.text}")
                                except Exception as e:
                                    st.error(f"Error connecting to API: {str(e)}")
                    
                    st.markdown("---")
    
    # Tab 5: Share Job Post
    with job_tabs[4]:
        st.subheader("Create Shareable Job Post Link")
        
        job_info = get_job_requirements()
        
        if not job_info.get('filename'):
            st.warning("No active job post found. Please upload or create a job post first.")
        else:
            st.info(f"Currently sharing: **{job_info.get('filename')}**")
            
            # Generate a unique link for this job post
            job_id = job_info.get('filename', '').split('.')[0]  # Use filename without extension as ID
            
            # In a real application, you would store this in a database
            # For now, we'll just create a shareable link with the job ID
            share_url = f"{API_URL}/apply/{job_id}"
            
            st.markdown("### Shareable Link")
            st.code(share_url, language="text")
            
            # Preview section
            st.markdown("### Application Preview")
            st.info("This is how candidates will see your job post when they use the link.")
            
            with st.expander("Job Post Preview", expanded=True):
                st.markdown(f"## {job_info.get('filename').split('_')[0].replace('_', ' ')}")
                st.markdown(job_info.get('requirements', 'Job description not available'))
                
                # Mock application form
                st.markdown("### Apply for this position")
                with st.form("mock_application_form"):
                    st.text_input("Full Name", placeholder="John Doe")
                    st.text_input("Email", placeholder="john.doe@example.com")
                    st.text_input("Phone", placeholder="+1 234 567 8900")
                    st.file_uploader("Upload your CV", type="pdf")
                    st.text_area("Cover Letter", placeholder="Why are you interested in this position?")
                    st.form_submit_button("Submit Application", disabled=True)
                
                st.info("This is a preview. The actual application form will be functional when candidates use the shareable link.")
            
            # Analytics section
            st.markdown("### Application Analytics")
            st.info("Track how many people have viewed and applied to your job post.")
            
            try:
                # Get real stats from the API
                response = requests.get(f"{API_URL}/job-stats/{job_id}")
                if response.status_code == 200:
                    stats = response.json()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Views", stats.get("views", 0))
                    with col2:
                        st.metric("Applications", stats.get("applications", 0))
                    with col3:
                        st.metric("Conversion Rate", f"{stats.get('conversion_rate', 0)}%")
                    
                    # Show applications if there are any
                    if stats.get("applications", 0) > 0:
                        st.markdown("### Recent Applications")
                        try:
                            app_response = requests.get(f"{API_URL}/applications/{job_id}")
                            if app_response.status_code == 200:
                                applications = app_response.json().get("applications", [])
                                for app in applications:
                                    with st.expander(f"{app.get('applicant_name')} - {app.get('submission_date', '')[:10]}"):
                                        st.write(f"**Email:** {app.get('email')}")
                                        st.write(f"**Phone:** {app.get('phone')}")
                                        if app.get('cover_letter'):
                                            st.write(f"**Cover Letter:** {app.get('cover_letter')}")
                                        st.write(f"**CV Filename:** {app.get('cv_filename')}")
                        except Exception as e:
                            st.error(f"Error loading applications: {str(e)}")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Views", "0")
                    with col2:
                        st.metric("Applications", "0")
                    with col3:
                        st.metric("Conversion Rate", "0%")
            except Exception as e:
                st.error(f"Error loading job statistics: {str(e)}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Views", "0")
                with col2:
                    st.metric("Applications", "0")
                with col3:
                    st.metric("Conversion Rate", "0%")

# Upload CV Page
elif page == "Upload CV":
    st.header("Upload New CV")
    
    # Add tips and requirements for CV uploads
    st.info("📌 **Tips for successful CV uploads:**\n"
            "- Make sure the PDF contains readable text (not scanned images)\n"
            "- The file should be less than 10MB\n"
            "- PDF should not be password protected\n"
            "- Ensure the content is relevant to job applications")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Show a preview of the file name and size
        file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 1)
        st.write(f"Selected: **{uploaded_file.name}** ({file_size_kb} KB)")
        
        # Add upload button with additional explanation
        with st.form("upload_cv_form"):
            st.write("Click 'Upload CV' to process and add this CV to the database.")
            submitted = st.form_submit_button("Upload CV")
            if submitted:
                # Show a progress indicator
                with st.spinner("Processing CV... This may take a few seconds."):
                    try:
                        files = {"file": uploaded_file}
                        response = requests.post(f"{API_URL}/upload-cv", files=files)
                        if response.status_code == 200:
                            st.success(f"✅ CV uploaded successfully!")
                            # Refresh the candidates list
                            load_candidates()
                        else:
                            error_message = "Unknown error"
                            try:
                                error_message = response.json().get("detail", "Unknown error")
                            except:
                                pass
                            st.error(f"❌ Error: {error_message}")
                            # Show troubleshooting tips for common errors
                            st.info("**Troubleshooting tips:**\n"
                                  "- Ensure the PDF contains selectable text\n"
                                  "- Try a different PDF with more content\n"
                                  "- Make sure the file isn't corrupted")
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")

# Add footer
st.sidebar.markdown("---")
st.sidebar.caption("CV Chatbot v1.0")

# Run the Streamlit app
if __name__ == "__main__":
    pass  # Streamlit runs the script directly
