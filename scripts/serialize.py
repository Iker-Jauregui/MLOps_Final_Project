#!/usr/bin/env python
# coding: utf-8

# # Serialize Best Model

# ## Imports

# In[36]:


import os
import json
import shutil
import pandas as pd
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

import onnxruntime as ort


# Set MLflow tracking URI
# Check if running in CI/CD with DagsHub
if os.environ.get('GITHUB_ACTIONS') or os.environ.get('DAGSHUB_USERNAME'):
    DAGSHUB_USERNAME = os.environ.get('DAGSHUB_USERNAME')
    DAGSHUB_REPO = 'my-first-repo'
    DAGSHUB_TOKEN = os.environ.get('DAGSHUB_TOKEN')
    
    if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
        mlflow_tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
        
        # Set credentials
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"Using DagsHub MLflow: {mlflow_tracking_uri}")
    else:
        print("Warning: DAGSHUB credentials not found, using local mlruns")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
else:
    # Local development - use local mlruns
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    print(f"Using local MLflow: {mlflow.get_tracking_uri()}")

# ## Load Best Model

# In[31]:


# Initialize MLflow client
client = MlflowClient()

# Set experiment name
EXPERIMENT_NAME = "revenue_prediction_training"

# Get experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found!")

experiment_id = experiment.experiment_id

# Search for all runs, ordered by test_rmse (ascending = best first)
all_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.test_rmse > 0",  # Only runs with test_rmse
    order_by=["metrics.test_rmse ASC"],
    max_results=1
)

if len(all_runs) == 0:
    raise ValueError("No runs found with test_rmse metric!")

# Get the best run
best_run = all_runs[0]
best_run_id = best_run.info.run_id

print(f"Best Run Found!")
print(f"Run ID: {best_run_id}")
print(f"Test RMSE: {best_run.data.metrics.get('test_rmse', 'N/A'):.4f}")
print(f"Validation RMSE: {best_run.data.metrics.get('val_rmse_cv', 'N/A'):.4f}")
print(f"\nBest Hyperparameters:")
for param, value in best_run.data.params.items():
    if param in ['n_estimators', 'max_depth', 'random_state']:
        print(f"  {param}: {value}")


# In[32]:


# Load the best model from MLflow
best_model_uri = f"runs:/{best_run_id}/random_forest_model"

try:
    best_model = mlflow.sklearn.load_model(best_model_uri)
    print(f"Loaded best model from run {best_run_id}")
    print(f"Model type: {type(best_model).__name__}")
    print(f"Number of trees: {best_model.n_estimators}")
    print(f"Max depth: {best_model.max_depth}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


# ## Export to ONNX

# In[42]:


# Define input type for ONNX (must match number of features)
n_features = best_model.n_features_in_
column_names = best_model.feature_names_in_
initial_type = [('float_input', FloatTensorType([None, n_features]))]

# Convert to ONNX
try:
    onnx_model = convert_sklearn(
        best_model,
        initial_types=initial_type,
        target_opset=18
    )
    print("Model converted to ONNX format")
except Exception as e:
    print(f"Error converting to ONNX: {e}")
    raise

# Create directory for ONNX models
onnx_dir = "models/onnx"
os.makedirs(onnx_dir, exist_ok=True)

# Save ONNX model
onnx_model_path = os.path.join(onnx_dir, f"best_rf.onnx")

with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

model_size_kb = os.path.getsize(onnx_model_path) / 1024

print(f"ONNX model saved to: {onnx_model_path}")
print(f"Model size: {model_size_kb:.2f} KB")


# Copy categorical metadata from MLflow artifacts
print("\n" + "="*60)
print("Copying categorical metadata...")
print("="*60)

try:
    # Download the artifacts directory for the best run
    artifacts_path = client.download_artifacts(best_run_id, "")
    
    # Path to dataset_metadata directory in MLflow artifacts
    metadata_dir = os.path.join(artifacts_path, "dataset_metadata")
    
    # Destination path (renamed to categorical_metadata.json)
    json_destination = os.path.join(onnx_dir, "categorical_metadata.json")
    
    if os.path.exists(metadata_dir) and os.path.isdir(metadata_dir):
        # Find the first JSON file in the dataset_metadata directory
        json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        
        if json_files:
            # Use the first JSON file found
            source_json = os.path.join(metadata_dir, json_files[0])
            shutil.copy2(source_json, json_destination)
            print(f"Categorical metadata copied from: {json_files[0]}")
            print(f"Saved as: {json_destination}")
            
            # Display metadata content
            with open(json_destination, 'r') as f:
                metadata = json.load(f)
            print(f"\nMetadata summary:")
            print(f"  ISRC classes: {metadata.get('ISRC', {}).get('n_classes', 'N/A')}")
            print(f"  Continent classes: {metadata.get('continent', {}).get('n_classes', 'N/A')}")
            print(f"  Zone classes: {metadata.get('zone', {}).get('n_classes', 'N/A')}")
        else:
            raise FileNotFoundError(f"No JSON files found in {metadata_dir}")
    else:
        print(f"Warning: dataset_metadata directory not found at {metadata_dir}")
        print(f"Searching recursively in artifacts directory...")
        
        # Fallback: search recursively for any JSON file
        found = False
        for root, dirs, files in os.walk(artifacts_path):
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                source_json = os.path.join(root, json_files[0])
                shutil.copy2(source_json, json_destination)
                print(f"Found and copied from: {source_json}")
                found = True
                break
        
        if not found:
            raise FileNotFoundError("No JSON metadata file found in MLflow run artifacts")

except Exception as e:
    print(f"Error copying metadata: {e}")
    raise

print("\n" + "="*60)
print("Serialization complete!")
print("="*60)
print(f"Output directory: {onnx_dir}")
print(f"  - ONNX model: best_rf.onnx")
print(f"  - Metadata: categorical_metadata.json")
