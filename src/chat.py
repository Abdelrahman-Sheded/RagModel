from config import DEPLOYMENT_NAME, AZURE_CONFIG
from langchain_openai import AzureChatOpenAI
from .ranking import truncate_text, rank_cvs
from .cv_management import add_cv, remove_cv_from_system
import re


def generate_response(user_prompt: str, faiss_index, metadata, job_desc_path: str) -> str:
    """Generates an LLM response given a user prompt."""
    
    def build_system_context(ranked_cvs, top_n=10):
        candidate_summaries = []
        for i, cv in enumerate(ranked_cvs[:top_n]):
            summary_text = cv.get('summary') or cv.get('cleaned_text', 'No summary available')

            summary_parts = [
                f"Candidate {i+1} (Rank #{i+1}):",
                f"- Filename: {cv['filename']}",
                f"- Email: {cv['contact'].get('email', 'N/A')}",
                f"- Phone: {cv['contact'].get('phone', 'N/A')}",
                f"- Similarity Score: {cv.get('similarity', 'N/A')}",
                f"- Summary: {truncate_text(summary_text, 500)}",
                "----------------------------------------"
            ]
            candidate_summaries.append("\n".join(summary_parts))
        
        return (
            "You are an AI assistant helping recruiters.\n"
            "Below is a summary of top candidates:\n\n" +
            "\n".join(candidate_summaries)
        )


    # 1. Rank candidates
    ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)

    # 2. Build system context
    system_context = build_system_context(ranked_cvs)

    # 3. Create model instance
    model = AzureChatOpenAI(
        azure_endpoint=AZURE_CONFIG["azure_endpoint"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        deployment_name=DEPLOYMENT_NAME,
        temperature=0.3
    )

    # 4. Construct message history
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_prompt}
    ]

    # 5. Get response
    response = model.invoke(messages)
    return response.content

# --- Interactive Chat Function with Streaming ---
# def interactive_chat(faiss_index, metadata, job_desc_path):
#     """
#     A free-form chat function that also:
#     - Summarizes the top candidates in a system message
#     - Lets the user add or remove CVs
#     - When the user asks 'info about #N', injects the full CV text for that candidate
#     """
#     def build_system_context(ranked_cvs, top_n=20):
#         """
#         Builds a system message summarizing the top candidates
#         so that the LLM can reference them in conversation.
#         Uses CV sections and chunks for more detailed information.
#         """
#         top_cvs = ranked_cvs[:top_n]
#         candidate_summaries = []
        
#         # Add explicit ranking explanation
#         ranking_explanation = (
#             "IMPORTANT RANKING INFORMATION:\n"
#             "The candidates below are numbered by their ranking. Candidate 1 is the best match for the job description,\n"
#             "Candidate 2 is the second best, and so on. Lower numbered candidates have been determined\n"
#             "to be better matches than higher numbered candidates. When comparing candidates with different\n"
#             "numbers, candidates with lower numbers should generally be considered better matches, unless\n"
#             "there is specific information in their profile that suggests otherwise.\n\n"
#         )
        
#         # Add instructions for handling missing information
#         missing_info_guidelines = (
#             "RESPONSE GUIDELINES FOR MISSING INFORMATION:\n"
#             "- Never respond with phrases like 'No information provided' or 'No information available'\n"
#             "- If specific information isn't explicitly mentioned in a candidate's profile, make reasonable inferences based on their background\n"
#             "- Use phrases like 'While not explicitly mentioned in their CV, based on their experience in [field]...' or\n"
#             "  'Their CV doesn't highlight this specific aspect, but given their role as [position], they likely...'\n"
#             "- Always provide a helpful response that offers insights based on available information\n"
#             "- If asked about very specific details not in the CV, acknowledge the limitation briefly and pivot to related information that IS available\n\n"
#         )
        
#         # Add specific guidelines for comparing candidates
#         comparison_guidelines = (
#             "COMPARISON GUIDELINES:\n"
#             "- When comparing candidates, focus only on information that is available for both\n"
#             "- Never create sections like 'Soft Skills: No specific soft skills mentioned' or 'Additional Qualifications: No specific qualifications mentioned'\n"
#             "- If a particular aspect (like soft skills, certifications, etc.) is not mentioned for a candidate, simply omit that section entirely from your comparison\n"
#             "- Structure your comparisons around the strengths and relevant experience that are actually present in the CVs\n"
#             "- Only mention the most relevant aspects for the job rather than trying to cover every possible category\n\n"
#         )
        
#         for i, cv in enumerate(top_cvs):
#             # Start with basic candidate info including clear ranking position
#             summary_parts = [
#                 f"Candidate {i+1} (Rank #{i+1}):",
#                 f"- Filename: {cv['filename']}",
#                 f"- Email: {cv['contact'].get('email', 'N/A')}",
#                 f"- Phone: {cv['contact'].get('phone', 'N/A')}",
#                 f"- Similarity Score: {cv.get('similarity', 'N/A')}"
#             ]
            
#             # Add relevant sections if available
#             if "sections" in cv:
#                 if "education" in cv["sections"]:
#                     summary_parts.append(f"- Education: {truncate_text(cv['sections']['education'], 500)}")
#                 if "experience" in cv["sections"]:
#                     summary_parts.append(f"- Experience: {truncate_text(cv['sections']['experience'], 800)}")
#                 if "skills" in cv["sections"]:
#                     summary_parts.append(f"- Skills: {truncate_text(cv['sections']['skills'], 500)}")
            
#             # If no sections, add some chunks
#             if "sections" not in cv or len(cv["sections"]) == 0:
#                 if "chunks" in cv and cv["chunks"]:
#                     summary_parts.append(f"- Profile Highlights:")
#                     for j, chunk in enumerate(cv["chunks"][:2]):
#                         summary_parts.append(f"  Excerpt {j+1}: {truncate_text(chunk, 400)}")
#                 else:
#                     # Fallback to cleaned text
#                     summary_parts.append(f"- Profile: {truncate_text(cv['cleaned_text'], 1000)}")
            
#             summary_parts.append("----------------------------------------")
#             summary = "\n".join(summary_parts)
#             candidate_summaries.append(summary)

#         system_context = (
#             "Below is a summary of the top candidates for the given job description.\n"
#             "These candidates were ranked by an AI system based on their match to the job requirements.\n" +
#             ranking_explanation +
#             missing_info_guidelines +
#             comparison_guidelines +
#             "\n".join(candidate_summaries)
#         )
#         return system_context

#     # Replace old AzureOpenAI client with LangChain's AzureChatOpenAI with updated parameters
#     chat_model = AzureChatOpenAI(
#         azure_endpoint=AZURE_CONFIG["azure_endpoint"],
#         api_key=AZURE_CONFIG["api_key"],
#         api_version=AZURE_CONFIG["api_version"],
#         deployment_name=DEPLOYMENT_NAME,
#         temperature=0.3
#     )

#     # 1) Initially rank CVs for the job description.
#     ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
#     system_context = build_system_context(ranked_cvs)
    
#     print("\nWelcome to the CV Chatbot!")
#     print("You can ask questions about the candidates, or type 'exit' to quit.")
#     print("Special commands:")
#     print("- 'info about #N': Get detailed information about candidate N")
#     print("- 'add cv <path>': Add a new CV to the system")
#     print("- 'remove cv <filename>': Remove a CV from the system")
#     print("- 'compare #N #M': Compare two candidates")
    
#     chat_history = []
    
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() == 'exit':
#             break
            
#         # Handle special commands
#         if user_input.lower().startswith('info about #'):
#             try:
#                 candidate_num = int(user_input.split('#')[1].strip()) - 1
#                 if 0 <= candidate_num < len(ranked_cvs):
#                     cv = ranked_cvs[candidate_num]
#                     print(f"\nDetailed information for Candidate #{candidate_num+1}:")
#                     print(f"Filename: {cv['filename']}")
#                     print(f"Email: {cv['contact'].get('email', 'N/A')}")
#                     print(f"Phone: {cv['contact'].get('phone', 'N/A')}")
#                     print(f"\nFull CV Text:\n{cv['raw_text'][:3000]}...")
#                     continue
#                 else:
#                     print(f"Invalid candidate number. Please choose between 1 and {len(ranked_cvs)}")
#                     continue
#             except:
#                 pass  # Fall through to normal chat if parsing fails
                
#         elif user_input.lower().startswith('add cv '):
#             path = user_input[7:].strip()
#             try:
#                 success = add_cv(path, faiss_index, metadata)
#                 if success:
#                     print(f"CV added successfully. Re-ranking candidates...")
#                     ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
#                     system_context = build_system_context(ranked_cvs)
#                 continue
#             except Exception as e:
#                 print(f"Error adding CV: {str(e)}")
#                 continue
                
#         elif user_input.lower().startswith('remove cv '):
#             filename = user_input[10:].strip()
#             try:
#                 updated_index, updated_metadata = remove_cv_from_system(filename, faiss_index, metadata)
#                 # Check if removal was successful by comparing metadata length
#                 if len(updated_metadata) < len(metadata):
#                     faiss_index, metadata = updated_index, updated_metadata
#                     print(f"CV removed successfully. Re-ranking candidates...")
#                     ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
#                     system_context = build_system_context(ranked_cvs)
#                 else:
#                     print(f"CV {filename} not found in the system")
#                 continue
#             except Exception as e:
#                 print(f"Error removing CV: {str(e)}")
#                 continue
                
#         elif user_input.lower().startswith('compare #'):
#             parts = user_input.split('#')
#             if len(parts) >= 3:
#                 try:
#                     candidate1 = int(parts[1].strip()) - 1
#                     candidate2 = int(parts[2].strip()) - 1
#                     if 0 <= candidate1 < len(ranked_cvs) and 0 <= candidate2 < len(ranked_cvs):
#                         compare_candidates(ranked_cvs[candidate1], ranked_cvs[candidate2], job_desc_path)
#                         continue
#                     else:
#                         print(f"Invalid candidate numbers. Please choose between 1 and {len(ranked_cvs)}")
#                         continue
#                 except:
#                     pass  # Fall through to normal chat if parsing fails
        
#         # Normal chat interaction
#         messages = [
#             {"role": "system", "content": system_context},
#         ]
        
#         # Add chat history
#         for message in chat_history:
#             messages.append(message)
            
#         # Add current user message
#         messages.append({"role": "user", "content": user_input})
        
#         # Get response from the model
#         response = chat_model.invoke(messages)
        
#         # Print the response
#         print(f"\nChatbot: {response.content}")
        
#         # Update chat history
#         chat_history.append({"role": "user", "content": user_input})
#         chat_history.append({"role": "assistant", "content": response.content})
        
#         # Keep chat history manageable
#         if len(chat_history) > 10:
#             chat_history = chat_history[-10:]

# Function to compare two candidates
def compare_candidates(cv1, cv2, job_desc_path):
    """Compare two candidates against the job description"""
    from .text_processing import extract_text_from_pdf, clean_text
    
    # Get job description
    raw_jd = extract_text_from_pdf(job_desc_path)
    cleaned_jd = clean_text(raw_jd)
    
    # Create comparison prompt
    prompt = f"""Compare these two candidates for the following job position:

Job Requirements:
{truncate_text(cleaned_jd, 800)}

Candidate 1: {cv1['filename']}
{truncate_text(cv1['cleaned_text'], 500)}

Candidate 2: {cv2['filename']}
{truncate_text(cv2['cleaned_text'], 500)}

COMPARISON FORMAT INSTRUCTIONS:
- Write your response as a flowing, conversational analysis, not as a structured comparison with rigid sections
- DO NOT use headers like "Education:", "Technical Skills:", "Soft Skills:", etc.
- DO NOT mention missing information - only discuss what is actually present in their profiles
- Begin with a brief introduction of both candidates and their overall fit
- Then directly compare their qualifications in context of the job requirements
- Focus on their strengths and relevant experience in a natural, paragraph-based format
- Conclude with your assessment of which candidate better matches the job requirements and why
- Your response should read like an expert recruiter's analysis, not like a form or template

Please compare their qualifications relative to the job requirements and determine which candidate is a better fit and why."""
    
    # Create a new chat model instance
    comparison_model = AzureChatOpenAI(
        azure_endpoint=AZURE_CONFIG["azure_endpoint"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        deployment_name=DEPLOYMENT_NAME,
        temperature=0.3
    )
    
    # Get comparison
    response = comparison_model.invoke(prompt)
    
    # Print the comparison
    print(f"\nComparison between {cv1['filename']} and {cv2['filename']}:")
    print(response.content)
    
    return response.content