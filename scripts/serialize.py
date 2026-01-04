#!/usr/bin/env python
# coding: utf-8

"""Serialize Best Model from MLflow Experiments"""

import os
import json
import shutil
import pickle
import pandas as pd
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort


# Configure MLflow Tracking
if os.environ.get('GITHUB_ACTIONS') or os.environ.get('DAGSHUB_USERNAME'):
    DAGSHUB_USERNAME = os.environ.get('DAGSHUB_USERNAME')
    DAGSHUB_REPO = 'my-first-repo'
    DAGSHUB_TOKEN = os.environ.get('DAGSHUB_TOKEN')
    
    if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
        mlflow_tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
        
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"Using DagsHub MLflow: {mlflow_tracking_uri}")
    else:
        print("Warning: DAGSHUB credentials not found, using local mlruns")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
else:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    print(f"Using local MLflow: {mlflow.get_tracking_uri()}")


# Initialize MLflow client
client = MlflowClient()

# Set experiment name
EXPERIMENT_NAME = "revenue_prediction_training"

# Get experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found!")

experiment_id = experiment.experiment_id

# Search for best run
all_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.val_rmse_cv > 0",
    order_by=["metrics.val_rmse_cv ASC"],
    max_results=1
)

if len(all_runs) == 0:
    raise ValueError("No runs found with val_rmse_cv metric!")

best_run = all_runs[0]
best_run_id = best_run.info.run_id

print(f"\n{'='*60}")
print("Best Run Found!")
print(f"{'='*60}")
print(f"Run ID: {best_run_id}")
print(f"Validation RMSE: {best_run.data.metrics.get('val_rmse_cv', 'N/A'):.4f}")
print(f"Test RMSE: {best_run.data.metrics.get('test_rmse', 'N/A'):.4f}")
print(f"\nBest Hyperparameters:")
for param, value in best_run.data.params.items():
    if param in ['n_estimators', 'max_depth', 'max_features', 'random_state']:
        print(f"  {param}: {value}")


# Load model from pickle artifact (works with DagsHub MLflow)
print(f"\n{'='*60}")
print("Loading model from artifacts...")
print(f"{'='*60}")

try:
    # Download artifacts directory
    artifacts_path = client.download_artifacts(best_run_id, "")
    print(f"Downloaded artifacts to: {artifacts_path}")
    
    # Look for pickle model in model_artifacts
    model_artifacts_dir = os.path.join(artifacts_path, "model_artifacts")
    
    if os.path.exists(model_artifacts_dir):
        # Find pickle file
        trial_number = best_run.data.params.get('trial_number', '0')
        pickle_files = [f for f in os.listdir(model_artifacts_dir) 
                       if f.endswith('.pkl')]
        
        if pickle_files:
            pickle_path = os.path.join(model_artifacts_dir, pickle_files[0])
            print(f"Loading model from: {pickle_path}")
            
            with open(pickle_path, 'rb') as f:
                best_model = pickle.load(f)
            
            print(f"Loaded best model from pickle")
            print(f"Model type: {type(best_model).__name__}")
            print(f"Number of trees: {best_model.n_estimators}")
            print(f"Max depth: {best_model.max_depth}")
        else:
            raise FileNotFoundError(f"No pickle files found in {model_artifacts_dir}")
    else:
        raise FileNotFoundError(f"model_artifacts directory not found at {artifacts_path}")
        
except Exception as e:
    print(f"Error loading model from artifacts: {e}")
    raise


# Export to ONNX
print(f"\n{'='*60}")
print("Converting to ONNX...")
print(f"{'='*60}")

n_features = best_model.n_features_in_
column_names = best_model.feature_names_in_
initial_type = [('float_input', FloatTensorType([None, n_features]))]

print(f"Model expects {n_features} features: {list(column_names)}")

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

# Save ONNX model
onnx_dir = "models/onnx"
os.makedirs(onnx_dir, exist_ok=True)

onnx_model_path = os.path.join(onnx_dir, "best_rf.onnx")

with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

model_size_kb = os.path.getsize(onnx_model_path) / 1024

print(f"ONNX model saved to: {onnx_model_path}")
print(f"Model size: {model_size_kb:.2f} KB")


# Copy categorical metadata
print(f"\n{'='*60}")
print("Copying categorical metadata...")
print(f"{'='*60}")

try:
    metadata_dir = os.path.join(artifacts_path, "dataset_metadata")
    json_destination = os.path.join(onnx_dir, "categorical_metadata.json")
    
    if os.path.exists(metadata_dir) and os.path.isdir(metadata_dir):
        json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        
        if json_files:
            source_json = os.path.join(metadata_dir, json_files[0])
            shutil.copy2(source_json, json_destination)
            print(f"Categorical metadata copied from: {json_files[0]}")
            print(f"Saved as: {json_destination}")
            
            with open(json_destination, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"\nMetadata summary:")
            if isinstance(metadata.get('ISRC'), list):
                print(f"  ISRC classes: {len(metadata.get('ISRC', []))}")
                print(f"  Continent classes: {len(metadata.get('continent', []))}")
                print(f"  Zone classes: {len(metadata.get('zone', []))}")
            else:
                print(f"  ISRC classes: {metadata.get('ISRC', {}).get('n_classes', 'N/A')}")
                print(f"  Continent classes: {metadata.get('continent', {}).get('n_classes', 'N/A')}")
                print(f"  Zone classes: {metadata.get('zone', {}).get('n_classes', 'N/A')}")
        else:
            raise FileNotFoundError(f"No JSON files found in {metadata_dir}")
    else:
        print(f"Warning: dataset_metadata not found, searching...")
        found = False
        for root, dirs, files in os.walk(artifacts_path):
            json_files = [f for f in files if f.endswith('.json') and 'hyperparams' not in f]
            if json_files:
                source_json = os.path.join(root, json_files[0])
                shutil.copy2(source_json, json_destination)
                print(f"Found and copied from: {source_json}")
                found = True
                break
        
        if not found:
            raise FileNotFoundError("No JSON metadata found in artifacts")

except Exception as e:
    print(f"Error copying metadata: {e}")
    raise

print(f"\n{'='*60}")
print("Serialization complete!")
print(f"{'='*60}")
print(f"Output directory: {onnx_dir}")
print(f"  - ONNX model: best_rf.onnx")
print(f"  - Metadata: categorical_metadata.json")