#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[26]:


import os
import json
import time
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import pickle

import mlflow
import mlflow.sklearn

import optuna
import optuna.visualization as viz
from optuna.integration.mlflow import MLflowCallback

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import root_mean_squared_error

import shap


# # Establish Development Iteration

# In[2]:


ITER = 1


# # Load train DataFrame

# ## Load DF and JSON

# In[3]:


# Get the project root directory (parent of scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'iter' + str(ITER), 'train')
df_train_file = list(filter(lambda x: 'parquet' in x and 'dvc' not in x, os.listdir(DF_TRAIN_DIR)))[0]
train_json_file = list(filter(lambda x: 'json' in x and 'dvc' not in x, os.listdir(DF_TRAIN_DIR)))[0]


# In[4]:


with open(os.path.join(DF_TRAIN_DIR, train_json_file), 'r') as f:
    train_json = json.load(f)

print(train_json['reporting_month'])


# In[5]:


# Read with PyArrow
table = pq.read_table(os.path.join(DF_TRAIN_DIR, df_train_file))

# Convert to pandas DataFrame
df_train = table.to_pandas()


# In[6]:


df_train.head()


# # Load test DataFrame

# In[7]:


DF_TEST_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'iter' + str(ITER), 'test')
df_test_file = list(filter(lambda x: 'parquet' in x and 'dvc' not in x, os.listdir(DF_TEST_DIR)))[0]


# In[8]:


# Read with PyArrow
table = pq.read_table(os.path.join(DF_TEST_DIR, df_test_file))

# Convert to pandas DataFrame
df_test = table.to_pandas()


# In[9]:


df_test.head()


# # Prepare DataFrames for training
# First, we will just drop "reporting_month" column, as we don't want to use it that information in our model. After that, we will check if there are unseen column values on test set. If so, we will replace them by the "UNKNOWN" word.

# ### Drop reporting_month

# In[10]:


df_train = df_train.drop(columns='reporting_month')
df_train.head()


# In[11]:


df_test = df_test.drop(columns='reporting_month')
df_test.head()


# ### Replace unseen values on Test set

# In[12]:


seen_isrc = set(train_json['ISRC'])
seen_zone = set(train_json['zone'])


# In[13]:


# Replace unseen values using pandas where (faster for large datasets)
df_test['ISRC'] = df_test['ISRC'].where(
    df_test['ISRC'].isin(seen_isrc), 
    'UNKNOWN'
)

df_test['zone'] = df_test['zone'].where(
    df_test['zone'].isin(seen_zone), 
    'UNKNOWN'
)


# In[14]:


# Check how many values were replaced
print(f"Unseen ISRCs replaced: {(df_test['ISRC'] == 'UNKNOWN').sum()}")
print(f"Unseen zones replaced: {(df_test['zone'] == 'UNKNOWN').sum()}")


# # Train Models

# ## MLFlow setup

# In[15]:


# Set MLflow tracking URI

# Check if running in CI/CD
if os.environ.get('GITHUB_ACTIONS'):
    # Use DagsHub's MLflow tracking server
    DAGSHUB_USERNAME = os.environ.get('DAGSHUB_USERNAME')
    DAGSHUB_REPO = 'my-first-repo'
    DAGSHUB_TOKEN = os.environ.get('DAGSHUB_TOKEN')
    
    mlflow_tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    
    # Set credentials
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"Using DagsHub MLflow: {mlflow_tracking_uri}")
else:
    # Local development - use local mlruns
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

# Set experiment name
EXPERIMENT_NAME = "revenue_prediction_training"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {EXPERIMENT_NAME}")


# ## Data splits setup

# In[16]:


TARGET_COLUMN = 'revenue'

X_train = df_train.drop(columns=[TARGET_COLUMN])
y_train = df_train[TARGET_COLUMN]

X_test = df_test.drop(columns=[TARGET_COLUMN])
y_test = df_test[TARGET_COLUMN]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Feature columns: {list(X_train.columns)}")


# ### Encode Categorical Features

# In[17]:


print(f"Available categorical columns: {list(train_json.keys())}\n")

# Apply transformations based on JSON category lists
for col in X_train.columns:
    if col in train_json:
        categories = train_json[col]

        # Add 'UNKNOWN' as index 0 if not already present
        if 'UNKNOWN' not in categories:
            categories = ['UNKNOWN'] + categories

        print(f"Encoding '{col}': {len(categories)} categories (including UNKNOWN)")

        # Create mapping from category to index
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        # Apply mapping to train and test
        X_train[col] = X_train[col].map(category_to_idx)
        X_test[col] = X_test[col].map(category_to_idx)

        # Check if there are still any NaN values
        if X_train[col].isna().any() or X_test[col].isna().any():
            n_unknown_train = X_train[col].isna().sum()
            n_unknown_test = X_test[col].isna().sum()
            print(f"  ERROR: Still have NaN values in '{col}' - Train: {n_unknown_train}, Test: {n_unknown_test}")
            X_train[col] = X_train[col].fillna(-1)
            X_test[col] = X_test[col].fillna(-1)
    else:
        print(f"'{col}' not in JSON (keeping as numeric)")

print(f"\nEncoded X_train shape: {X_train.shape}")
print(f"Encoded X_test shape: {X_test.shape}")
print("\nFirst few rows after encoding:")
print(X_train.head())
print("\nData types after encoding:")
print(X_train.dtypes)


# In[18]:


# Calculate number of CV folds so validation split equals test size
# df_train is 80% of total data, df_test is 20% of total data
# We want validation to also be 20% of total data

total_data_size = len(df_train) + len(df_test)
test_size = len(df_test)
train_size = len(df_train)

# Validation should be 20% of total = test_size
# If we use k-fold CV, each fold is (1/k) of df_train
# We want (1/k) * train_size = test_size
# So k = train_size / test_size

k_folds = int(train_size / test_size)

print(f"Total data size: {total_data_size}")
print(f"Train size: {train_size} ({train_size/total_data_size*100:.1f}%)")
print(f"Test size: {test_size} ({test_size/total_data_size*100:.1f}%)")
print(f"Using {k_folds}-fold CV")
print(f"Each validation fold: ~{train_size/k_folds:.0f} samples ({(train_size/k_folds)/total_data_size*100:.1f}%)")


# ## Optuna objective function

# In[19]:


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    Trains Random Forest with cross-validation and logs to MLflow.
    """

    # Suggest hyperparameters
    params = {
        # Tree ensemble parameters
        'n_estimators': trial.suggest_int('n_estimators', 10, 30),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        
        # Feature sampling
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7, 0.9, None]),
        
        # Fixed parameters
        'random_state': 42,
        'n_jobs': 2
    }
    
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}

    # Start MLflow run (nested under parent run)
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("cv_folds", k_folds)

        # Create model
        model = RandomForestRegressor(**params)

        # Train with cross-validation and track training time
        start_time = time.time()

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=k_folds,
            scoring='neg_root_mean_squared_error',
            return_train_score=True,
            n_jobs=2
        )

        training_time = time.time() - start_time

        # Calculate average metrics
        train_rmse = -cv_results['train_score'].mean()
        val_rmse = -cv_results['test_score'].mean()

        # Log CV metrics
        mlflow.log_metric("train_rmse_cv", train_rmse)
        mlflow.log_metric("val_rmse_cv", val_rmse)
        mlflow.log_metric("training_time_seconds", training_time)

        # Log per-fold metrics
        for fold_idx in range(k_folds):
            mlflow.log_metric(f"fold_{fold_idx}_train_rmse", 
                            -cv_results['train_score'][fold_idx])
            mlflow.log_metric(f"fold_{fold_idx}_val_rmse", 
                            -cv_results['test_score'][fold_idx])

        # Train final model on full training set
        model.fit(X_train, y_train)

        # Evaluate on test set
        eval_start_time = time.time()
        y_pred_test = model.predict(X_test)
        eval_time = time.time() - eval_start_time

        test_rmse = root_mean_squared_error(y_test, y_pred_test)

        # Log test metrics
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("evaluation_time_seconds", eval_time)

        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Save and log model as pickle
        model_path = f"models/trial_{trial.number}_model.pkl"
        os.makedirs("models", exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path, "model_artifacts")

        # Feature importance plot
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title(f'Top 10 Feature Importances - Trial {trial.number}')
        plt.tight_layout()
        importance_plot_path = f"plots/trial_{trial.number}_feature_importance.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(importance_plot_path, "feature_importance")

        # SHAP values (on a sample for speed in toy training)
        sample_size = min(100, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        shap_plot_path = f"plots/trial_{trial.number}_shap_summary.png"
        plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_plot_path, "shap_analysis")

        # SHAP feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        shap_importance_path = f"plots/trial_{trial.number}_shap_importance.png"
        plt.savefig(shap_importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_importance_path, "shap_analysis")

        # Log dataset metadata files (if they exist)
        dataset_dvc_path = os.path.join(DF_TRAIN_DIR, df_train_file + '.dvc')
        dataset_json_path = os.path.join(DF_TRAIN_DIR, train_json_file)

        if os.path.exists(dataset_dvc_path):
            mlflow.log_artifact(dataset_dvc_path, "dataset_metadata")

        if os.path.exists(dataset_json_path):
            mlflow.log_artifact(dataset_json_path, "dataset_metadata")

        # Create and log hyperparameters JSON
        hyperparams_dict = {
            **params,
            'cv_folds': k_folds,
            'trial_number': trial.number
        }
        hyperparams_path = f"configs/trial_{trial.number}_hyperparams.json"
        os.makedirs("configs", exist_ok=True)
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams_dict, f, indent=4)
        mlflow.log_artifact(hyperparams_path, "hyperparameters")

        print(f"Trial {trial.number}: val_rmse={val_rmse:.4f}, test_rmse={test_rmse:.4f}")

    # Return validation RMSE for Optuna to optimize
    return val_rmse


# ## Launch Optuna Study with MLFlow

# In[24]:


# Safety: end any active runs before starting
mlflow.end_run()

# Create parent MLflow run
with mlflow.start_run(run_name="optuna_optimization") as parent_run:

    # Log study configuration
    mlflow.log_param("optimization_metric", "val_rmse")
    mlflow.log_param("n_trials", 50)
    mlflow.log_param("model_type", "RandomForest")

    # Create Optuna study
    study = optuna.create_study(
        study_name="revenue_rf_optimization",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Run optimization
    print("Starting Optuna optimization...")
    study.optimize(
        objective,
        n_trials=50,
        show_progress_bar=True
    )

    # Log best trial information
    best_trial = study.best_trial
    mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
    mlflow.log_metric("best_val_rmse", best_trial.value)

    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    print(f"Best trial number: {best_trial.number}")
    print(f"Best validation RMSE: {best_trial.value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Generate and log optimization visualizations
    print("\nGenerating optimization visualizations...")
    
    try:
        # Create plots directory
        os.makedirs("plots", exist_ok=True)
        
        # Plot optimization history (save as HTML)
        fig1 = viz.plot_optimization_history(study)
        fig1.write_html("plots/optimization_history.html")
        print("Saved: plots/optimization_history.html")
        
        # Plot parameter importances
        fig2 = viz.plot_param_importances(study)
        fig2.write_html("plots/param_importances.html")
        print("Saved: plots/param_importances.html")
        
        # Plot parallel coordinate
        fig3 = viz.plot_parallel_coordinate(study)
        fig3.write_html("plots/parallel_coordinate.html")
        print("Saved: plots/parallel_coordinate.html")
        
        # Log to MLflow (we're already inside the run context)
        mlflow.log_artifact("plots/optimization_history.html", "optimization_plots")
        mlflow.log_artifact("plots/param_importances.html", "optimization_plots")
        mlflow.log_artifact("plots/parallel_coordinate.html", "optimization_plots")
        
        print("Logged optimization plots to MLflow")
        
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
