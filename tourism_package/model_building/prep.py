
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from datasets import Dataset
from huggingface_hub import HfApi, login

# --- Configuration ---
# Replace with your actual Hugging Face username
HF_USERNAME = "RajendrakumarPachaiappan"
HF_REPO_DATA = f"{HF_USERNAME}/tourism-package-dataset"
HF_REPO_MODEL = f"{HF_USERNAME}/tourism-package-prediction-model"

# Authenticate with Hugging Face (uses HF_TOKEN environment variable from GitHub Secrets)
try:
    login(token=os.getenv("HF_TOKEN"))
except Exception as e:
    print(f"Hugging Face login failed: {e}. Ensure HF_TOKEN is set.")

# --- File Paths ---
DATA_PATH = "hf://datasets/RajendrakumarPachaiappan/tourism-package-dataset/tourism.csv"
PREPROCESSOR_PATH = "preprocessor.joblib"
# Create artifacts directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)
LOCAL_PREPROCESSOR_FILE = os.path.join("artifacts", PREPROCESSOR_PATH)
TRAIN_FILE = os.path.join("artifacts", "tourism_train.csv")
TEST_FILE = os.path.join("artifacts", "tourism_test.csv")

# --- Data Loading and Cleaning ---
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Drop CustomerID as it's an identifier
df = df.drop(columns=['CustomerID'])

# Standardize some column values (e.g., 'Fe Male' to 'Female')
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
# The dataset has an unnamed index column; check if it exists and drop it
if df.columns[0] == 'Unnamed: 0':
    df = df.drop(columns=[df.columns[0]])

# --- Preprocessing Pipeline Definition ---

# Identify feature types
# Target variable
TARGET = 'ProdTaken'
# Columns for numerical imputation and scaling
NUM_FEATURES = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting',
    'MonthlyIncome', 'CityTier', 'Passport', 'OwnCar', 'PitchSatisfactionScore'
]
# Columns for categorical imputation and One-Hot Encoding
CAT_FEATURES = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation'
]

# Create preprocessing steps
numeric_transformer = Pipeline(steps=[
    # Impute numerical features with the median
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    # Impute categorical features with the most frequent value
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # One-hot encode the categorical features
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the main preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUM_FEATURES),
        ('cat', categorical_transformer, CAT_FEATURES)
    ],
    remainder='drop', # Drop any other columns not specified
    n_jobs=-1
)

# --- Apply Preprocessing, Split, and Save Artifacts ---

# Separate features (X) and target (y)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Fit the preprocessor on the entire dataset (for consistency in feature space)
print("Fitting preprocessor...")
preprocessor.fit(X)

# Split the original, raw data (recommended to keep original for MLOps data versioning)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Combine features and target for saving as CSV
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save the preprocessor locally
print(f"Saving fitted preprocessor to {LOCAL_PREPROCESSOR_FILE}...")
joblib.dump(preprocessor, LOCAL_PREPROCESSOR_FILE)

# Save raw train and test data locally
print(f"Saving raw train/test data to {TRAIN_FILE} and {TEST_FILE}...")
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# --- Register Artifacts on Hugging Face Hub ---

api = HfApi()

# 1. Upload Data to Hugging Face Dataset Hub
print(f"Uploading data to HF Dataset Hub: {HF_REPO_DATA}...")
# Convert to Hugging Face Dataset objects to ensure proper format for upload
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
hf_dataset_dict = {
    'train': train_dataset,
    'test': test_dataset
}
from datasets import DatasetDict
hf_dataset = DatasetDict(hf_dataset_dict)
hf_dataset.push_to_hub(HF_REPO_DATA, private=False)

print("Data uploaded successfully.")

# 2. Upload Preprocessor to Hugging Face Model Hub (for deployment)
print(f"Uploading preprocessor to HF Model Hub: {HF_REPO_MODEL}...")

# Check if repo exists, create if not (model hub)
try:
    api.create_repo(repo_id=HF_REPO_MODEL, repo_type="model", private=False)
except Exception:
    pass # Repo already exists

api.upload_file(
    path_or_fileobj=LOCAL_PREPROCESSOR_FILE,
    path_in_repo=PREPROCESSOR_PATH,
    repo_id=HF_REPO_MODEL,
    repo_type="model",
)
print("Preprocessor uploaded successfully.")

