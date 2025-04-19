from config import DEPLOYMENT_NAME, AZURE_CONFIG
from langchain_openai import AzureChatOpenAI

def generate_cv_summary(text):
    model = AzureChatOpenAI(
        azure_endpoint=AZURE_CONFIG["azure_endpoint"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        deployment_name=DEPLOYMENT_NAME,
        temperature=0.3
    )
    prompt = f"Summarize the following candidate CV:\n\n{text[:2000]}"
    return model.invoke(prompt).content
