import os
from huggingface_hub import login

def get_hf_token():
    """
    Fetch Hugging Face token from:
    - Environment variable: HF_TOKEN
    - Fallback to manual input if not found
    """
    token = os.environ.get("HF_TOKEN")

    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except ImportError:
            pass

    if not token:
        token = input("Enter your Hugging Face token: ").strip()

    return token

def authenticate_huggingface():
    """
    Log in to Hugging Face Hub using a securely retrieved token
    """
    hf_token = get_hf_token()
    if hf_token:
        print("Logging into Hugging Face...")
        login(token=hf_token)
    else:
        print("No Hugging Face token provided. Authentication skipped.")

if __name__ == "__main__":
    authenticate_huggingface()
