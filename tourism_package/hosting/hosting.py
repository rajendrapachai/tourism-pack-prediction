import os
from huggingface_hub import HfApi

# --- Configuration ---
# Replace with your actual Hugging Face Space ID
HF_SPACE_ID = "RajendrakumarPachaiappan/tourism-package-prediction"
DEPLOYMENT_FOLDER = "tourism_package/deployment"

# Authenticate with Hugging Face (uses HF_TOKEN environment variable)
# Ensure the HF_TOKEN has WRITE permission!
try:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set.")
    api = HfApi(token=HF_TOKEN)
except Exception as e:
    print(f"Hugging Face API initialization failed: {e}")
    exit(1)

# Check if the Space repo exists, create it if necessary (Crucial for CI/CD)
try:
    print(f"Attempting to create or verify Hugging Face Space: {HF_SPACE_ID}...")
    api.create_repo(
        repo_id=HF_SPACE_ID,
        repo_type="space",
        space_sdk="streamlit",  # Specify the SDK as Streamlit
        private=False            # Ensure the space is public
    )
    print("Space repository verified/created.")
except Exception as e:
    print(f"Error creating/verifying space: {e}")
    # Continue upload attempt even on potential error, as it might just be a "repo already exists" exception

# Upload the deployment files to the root of the Hugging Face Space
print(f"Uploading deployment files from {DEPLOYMENT_FOLDER} to Hugging Face Space...")
try:
    api.upload_folder(
        folder_path=DEPLOYMENT_FOLDER,
        repo_id=HF_SPACE_ID,
        repo_type="space",
        path_in_repo="/", # Uploads files to the root of the Space
        commit_message="CI/CD: Automated Streamlit app update"
    )
    print(f"Deployment to Hugging Face Space '{HF_SPACE_ID}' complete.")
except Exception as e:
    print(f"Failed to upload deployment folder: {e}")
    exit(1)
