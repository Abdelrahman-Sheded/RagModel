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

def interactive_chat(faiss_index, metadata, job_desc_path):
    """
    A free-form chat function that also:
    - Summarizes the top candidates in a system message
    - Lets the user add or remove CVs
    - When the user asks 'info about #N', injects the full CV text for that candidate
    """
    def build_system_context(ranked_cvs, top_n=20):
        """
        Builds a system message summarizing the top candidates
        so that the LLM can reference them in conversation.
        Uses CV sections and chunks for more detailed information.
        """
        top_cvs = ranked_cvs[:top_n]
        candidate_summaries = []
        
        # Add explicit ranking explanation
        ranking_explanation = (
            "IMPORTANT RANKING INFORMATION:\n"
            "The candidates below are numbered by their ranking. Candidate 1 is the best match for the job description,\n"
            "Candidate 2 is the second best, and so on. Lower numbered candidates have been determined\n"
            "to be better matches than higher numbered candidates. When comparing candidates with different\n"
            "numbers, candidates with lower numbers should generally be considered better matches, unless\n"
            "there is specific information in their profile that suggests otherwise.\n\n"
        )
        
        # Add instructions for handling missing information
        missing_info_guidelines = (
            "RESPONSE GUIDELINES FOR MISSING INFORMATION:\n"
            "- Never respond with phrases like 'No information provided' or 'No information available'\n"
            "- If specific information isn't explicitly mentioned in a candidate's profile, make reasonable inferences based on their background\n"
            "- Use phrases like 'While not explicitly mentioned in their CV, based on their experience in [field]...' or\n"
            "  'Their CV doesn't highlight this specific aspect, but given their role as [position], they likely...'\n"
            "- Always provide a helpful response that offers insights based on available information\n"
            "- If asked about very specific details not in the CV, acknowledge the limitation briefly and pivot to related information that IS available\n\n"
        )
        
        # Add specific guidelines for comparing candidates
        comparison_guidelines = (
            "COMPARISON GUIDELINES:\n"
            "- When comparing candidates, focus only on information that is available for both\n"
            "- Never create sections like 'Soft Skills: No specific soft skills mentioned' or 'Additional Qualifications: No specific qualifications mentioned'\n"
            "- If a particular aspect (like soft skills, certifications, etc.) is not mentioned for a candidate, simply omit that section entirely from your comparison\n"
            "- Structure your comparisons around the strengths and relevant experience that are actually present in the CVs\n"
            "- Only mention the most relevant aspects for the job rather than trying to cover every possible category\n\n"
        )
        
        for i, cv in enumerate(top_cvs):
            # Start with basic candidate info including clear ranking position
            summary_parts = [
                f"Candidate {i+1} (Rank #{i+1}):",
                f"- Filename: {cv['filename']}",
                f"- Email: {cv['contact'].get('email', 'N/A')}",
                f"- Phone: {cv['contact'].get('phone', 'N/A')}",
                f"- Similarity Score: {cv.get('similarity', 'N/A')}"
            ]
            
            # Add relevant sections if available
            if "sections" in cv:
                if "education" in cv["sections"]:
                    summary_parts.append(f"- Education: {truncate_text(cv['sections']['education'], 500)}")
                if "experience" in cv["sections"]:
                    summary_parts.append(f"- Experience: {truncate_text(cv['sections']['experience'], 800)}")
                if "skills" in cv["sections"]:
                    summary_parts.append(f"- Skills: {truncate_text(cv['sections']['skills'], 500)}")
            
            # If no sections, add some chunks
            if "sections" not in cv or len(cv["sections"]) == 0:
                if "chunks" in cv and cv["chunks"]:
                    summary_parts.append(f"- Profile Highlights:")
                    for j, chunk in enumerate(cv["chunks"][:2]):
                        summary_parts.append(f"  Excerpt {j+1}: {truncate_text(chunk, 400)}")
                else:
                    # Fallback to cleaned text
                    summary_parts.append(f"- Profile: {truncate_text(cv['cleaned_text'], 1000)}")
            
            summary_parts.append("----------------------------------------")
            summary = "\n".join(summary_parts)
            candidate_summaries.append(summary)

        system_context = (
            "Below is a summary of the top candidates for the given job description.\n"
            "These candidates were ranked by an AI system based on their match to the job requirements.\n" +
            ranking_explanation +
            missing_info_guidelines +
            comparison_guidelines +
            "\n".join(candidate_summaries)
        )
        return system_context

    # Replace old AzureOpenAI client with LangChain's AzureChatOpenAI with updated parameters
    chat_model = AzureChatOpenAI(
        azure_endpoint=AZURE_CONFIG["azure_endpoint"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        deployment_name=DEPLOYMENT_NAME,
        temperature=0.3
    )

    # 1) Initially rank CVs for the job description.
    ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
    system_context = build_system_context(ranked_cvs)
    
    print("\nWelcome to the CV Chatbot!")
    print("You can ask questions about the candidates, or type 'exit' to quit.")
    print("Special commands:")
    print("- 'info about #N': Get detailed information about candidate N")
    print("- 'add cv <path>': Add a new CV to the system")
    print("- 'remove cv <filename>': Remove a CV from the system")
    print("- 'compare #N #M': Compare two candidates")
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # Handle special commands
        if user_input.lower().startswith('info about #'):
            try:
                candidate_num = int(user_input.split('#')[1].strip()) - 1
                if 0 <= candidate_num < len(ranked_cvs):
                    cv = ranked_cvs[candidate_num]
                    print(f"\nDetailed information for Candidate #{candidate_num+1}:")
                    print(f"Filename: {cv['filename']}")
                    print(f"Email: {cv['contact'].get('email', 'N/A')}")
                    print(f"Phone: {cv['contact'].get('phone', 'N/A')}")
                    print(f"\nFull CV Text:\n{cv['raw_text'][:3000]}...")
                    continue
                else:
                    print(f"Invalid candidate number. Please choose between 1 and {len(ranked_cvs)}")
                    continue
            except:
                pass  # Fall through to normal chat if parsing fails
                
        elif user_input.lower().startswith('add cv '):
            path = user_input[7:].strip()
            try:
                success = add_cv(path, faiss_index, metadata)
                if success:
                    print(f"Successfully added CV: {path}")
                    # Re-rank CVs after adding new one
                    ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
                    system_context = build_system_context(ranked_cvs)
                else:
                    print(f"Failed to add CV: {path}")
                continue
            except Exception as e:
                print(f"Error adding CV: {str(e)}")
                continue
                
        elif user_input.lower().startswith('remove cv '):
            filename = user_input[10:].strip()
            try:
                success = remove_cv_from_system(filename, faiss_index, metadata)
                if success:
                    print(f"Successfully removed CV: {filename}")
                    # Re-rank CVs after removing one
                    ranked_cvs = rank_cvs(job_desc_path, faiss_index, metadata)
                    system_context = build_system_context(ranked_cvs)
                else:
                    print(f"Failed to remove CV: {filename}")
                continue
            except Exception as e:
                print(f"Error removing CV: {str(e)}")
                continue
                
        elif user_input.lower().startswith('compare #'):
            try:
                # Extract candidate numbers
                nums = re.findall(r'#(\d+)', user_input)
                if len(nums) == 2:
                    num1, num2 = int(nums[0]) - 1, int(nums[1]) - 1
                    if 0 <= num1 < len(ranked_cvs) and 0 <= num2 < len(ranked_cvs):
                        cv1, cv2 = ranked_cvs[num1], ranked_cvs[num2]
                        response = compare_candidates(cv1, cv2, job_desc_path)
                        print(f"\n{response}")
                        continue
                    else:
                        print(f"Invalid candidate numbers. Please choose between 1 and {len(ranked_cvs)}")
                        continue
            except:
                pass  # Fall through to normal chat if parsing fails
        
        # Normal chat interaction
        try:
            # Construct messages with history
            messages = [
                {"role": "system", "content": system_context}
            ]
            
            # Add chat history
            for msg in chat_history[-5:]:  # Keep last 5 messages for context
                messages.append(msg)
            
            # Add current user message
            messages.append({"role": "user", "content": user_input})
            
            # Get response
            response = chat_model.invoke(messages)
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response.content})
            
            print(f"\nAssistant: {response.content}")
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            continue

def compare_candidates(cv1, cv2, job_desc_path):
    """Compare two candidates and provide insights"""
    # Create model instance
    model = AzureChatOpenAI(
        azure_endpoint=AZURE_CONFIG["azure_endpoint"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        deployment_name=DEPLOYMENT_NAME,
        temperature=0.3
    )
    
    # Build comparison prompt
    comparison_prompt = f"""
    Compare these two candidates for the job description:
    
    Candidate 1:
    - Filename: {cv1['filename']}
    - Email: {cv1['contact'].get('email', 'N/A')}
    - Phone: {cv1['contact'].get('phone', 'N/A')}
    - Similarity Score: {cv1.get('similarity', 'N/A')}
    - Profile: {truncate_text(cv1.get('cleaned_text', ''), 1000)}
    
    Candidate 2:
    - Filename: {cv2['filename']}
    - Email: {cv2['contact'].get('email', 'N/A')}
    - Phone: {cv2['contact'].get('phone', 'N/A')}
    - Similarity Score: {cv2.get('similarity', 'N/A')}
    - Profile: {truncate_text(cv2.get('cleaned_text', ''), 1000)}
    
    Please provide a detailed comparison focusing on:
    1. Relevant experience and skills
    2. Education and qualifications
    3. Overall fit for the position
    4. Key strengths of each candidate
    5. Potential areas for development
    
    Format your response in a clear, structured way.
    """
    
    # Get comparison
    response = model.invoke([{"role": "user", "content": comparison_prompt}])
    return response.content