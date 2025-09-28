

import os
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, login
import mlflow

# --- Configuration ---

HF_USERNAME = "RajendrakumarPachaiappan"
HF_REPO_DATA = f"{HF_USERNAME}/tourism-package-dataset"
HF_REPO_MODEL = f"{HF_USERNAME}/tourism-package-prediction-model"

PREPROCESSOR_PATH = "preprocessor.joblib"
LOCAL_MODEL_FILE = os.path.join("artifacts", "model.joblib")

# Set up MLflow tracking (local file store)
MLFLOW_TRACKING_URI = "mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Tourism_Package_Prediction_Exp")

# Authenticate with Hugging Face (uses HF_TOKEN environment variable)
try:
    login(token=os.getenv("HF_TOKEN"))
except Exception as e:
    print(f"Hugging Face login failed: {e}. Ensure HF_TOKEN is set.")

# --- Utility Function ---
def fetch_artifact_from_hub(repo_id, filename, repo_type="model"):
    """Fetches an artifact from the Hugging Face Hub."""
    api = HfApi()
    temp_path = os.path.join("artifacts", filename)
    api.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder="",
        repo_type=repo_type,
        local_dir="artifacts" # Download to the artifacts folder
    )
    return temp_path

# --- Model Training Workflow ---
with mlflow.start_run():
    # Load Data from Hugging Face Dataset Hub
    print(f"Loading data from HF Dataset Hub: {HF_REPO_DATA}...")
    hf_data = load_dataset(HF_REPO_DATA)
    train_df = hf_data['train'].to_pandas()
    test_df = hf_data['test'].to_pandas()

    TARGET = 'ProdTaken'
    X_train_raw = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test_raw = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Load Preprocessor
    print(f"Loading preprocessor from HF Model Hub: {HF_REPO_MODEL}...")
    preprocessor_path = fetch_artifact_from_hub(HF_REPO_MODEL, PREPROCESSOR_PATH)
    preprocessor = joblib.load(preprocessor_path)

    # Transform Data
    print("Transforming data...")
    X_train_processed = preprocessor.transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # Define Model and Parameter Grid
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Log model type
    mlflow.log_param("model_type", "RandomForestClassifier")

    # Tune Model with Randomized Search
    print("Starting Randomized Search for hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='f1', # Use F1-score as target is likely imbalanced
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train_processed, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Log best parameters
    mlflow.log_params(best_params)
    print("Best Parameters:", best_params)

    # Evaluate Model Performance
    y_pred = best_model.predict(X_test_processed)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    # Log metrics
    mlflow.log_metrics(metrics)
    print("Model Metrics:", metrics)

    # Save and Register Best Model to Hugging Face Model Hub
    print(f"Saving best model locally to {LOCAL_MODEL_FILE}...")
    joblib.dump(best_model, LOCAL_MODEL_FILE)

    print(f"Uploading model to HF Model Hub: {HF_REPO_MODEL}...")
    api = HfApi()

    # Check if repo exists, create if not (model hub)
    try:
        api.create_repo(repo_id=HF_REPO_MODEL, repo_type="model", private=False)
    except Exception:
        pass # Repo already exists

    api.upload_file(
        path_or_fileobj=LOCAL_MODEL_FILE,
        path_in_repo="model.joblib",
        repo_id=HF_REPO_MODEL,
        repo_type="model",
    )
    print("Model uploaded successfully to Hugging Face Model Hub.")

    # Log model with MLflow (optional, but good practice)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="RandomForestTourismPredictor",
        signature=mlflow.models.infer_signature(X_train_processed, best_model.predict(X_train_processed))
    )
