#!/usr/bin/env python
# coding: utf-8

# # Serialize Best Model

# ## Imports

# In[36]:


import os
import json
import pandas as pd
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

import onnxruntime as ort


# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

# ## Load Best Model

# In[31]:


# Initialize MLflow client
client = MlflowClient()

# Set your experiment name (change this to match your experiment)
EXPERIMENT_NAME = "revenue_prediction_toy_training"

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

# Save ONNX model with run_id in filename for traceability
onnx_model_path = os.path.join(onnx_dir, f"best_rf_{best_run_id[:8]}.onnx")

with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

model_size_kb = os.path.getsize(onnx_model_path) / 1024

print(f"ONNX model saved to: {onnx_model_path}")
print(f"Model size: {model_size_kb:.2f} KB")


