from huggingface_hub import HfApi, login
import os

# --- CONFIGURATION ---
# 1. Get your token from: https://huggingface.co/settings/tokens
# 2. Paste it below OR set it as an environment variable HF_TOKEN
HF_TOKEN = "YOUR_HUGGINGFACE_WRITE_TOKEN_HERE" 

# 3. Set your repo ID (e.g., "YourUsername/ModelName")
REPO_ID = "maximaverick/LFM2.5-1.2B-Financial-Analyst-Thinking" 

# 4. Path to your model folder
MODEL_PATH = r"e:\LLM Tuning\finetune_project\model_output"
# ---------------------

def upload_model():
    print(f"Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    
    api = HfApi()
    
    print(f"Creating repository '{REPO_ID}' (if it doesn't exist)...")
    try:
        api.create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo creation/check failed: {e}")
        return

    print(f"Uploading files from {MODEL_PATH} to https://huggingface.co/{REPO_ID} ...")
    print("This may take a while depending on your upload speed.")
    
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        # Ignore safetensors, checkpoints, and backup folders
        ignore_patterns=["*.safetensors", ".ipynb_checkpoints/*", "checkpoint-*", "adapter_backup/*"], 
    )
    
    print("\n✅ Upload Complete!")
    print(f"View your model at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    # verify directory exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} does not exist.")
    else:
        upload_model()
