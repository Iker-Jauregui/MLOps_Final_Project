# MLOps Final Project: Music Track Revenue Prediction

[![CICD](https://github.com/Iker-Jauregui/MLOps_Final_Project/actions/workflows/CICD.yml/badge.svg)](https://github.com/Iker-Jauregui/MLOps_Final_Project/actions/workflows/CICD.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121.2+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning system for predicting music track revenue, showcasing MLOps best practices including automated CI/CD, experiment tracking, model versioning, and production monitoring.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the API](#running-the-api)
- [Development](#development)
  - [Training Models](#training-models)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
- [Deployment](#deployment)
  - [Docker](#docker)
  - [Monitoring](#monitoring)
- [API Documentation](#api-documentation)
- [CI/CD Pipeline](#cicd-pipeline)
- [License](#license)

## Overview

This project implements an end-to-end MLOps pipeline for predicting music track revenue using machine learning.  The system leverages Random Forest models with ONNX runtime for optimized inference in production. 

**Key Technologies:**
- **ML Framework**: scikit-learn (Random Forest)
- **Experiment Tracking**:  MLflow
- **Model Format**:  ONNX for production inference
- **API**: FastAPI with async support
- **Monitoring**:  Prometheus + Grafana
- **Data Versioning**: DVC
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## Features

- **FastAPI REST API** with automatic OpenAPI documentation
- **Experiment Tracking** using MLflow
- **Hyperparameter Optimization** with Optuna
- **Model Monitoring** via Prometheus + Grafana dashboards
- **Containerized Deployment** with Docker
- **Automated CI/CD** pipeline with GitHub Actions
- **Data Versioning** using DVC
- **ONNX Runtime** for fast inference
- **Comprehensive Testing** with pytest
- **Model Explainability** using SHAP values
- **Web UI** for interactive predictions

## Architecture

```
┌─────────────────┐
│  Data Sources   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│   DVC Storage   │◄─────┤  Data Pipeline   │
└─────────────────┘      └────────┬─────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Training with  │
                         │  Optuna/MLflow  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  ONNX Export    │
                         └────────┬────────┘
                                  │
                                  ▼
    ┌──────────────────────────────────────┐
    │         FastAPI Service              │
    │  ┌────────────┐  ┌────────────────┐  │
    │  │  REST API  <-->  Prometheus    │  │
    │  │            │  │  Metrics       │  │
    │  └────────────┘  └────────────────┘  │
    └──────────────────────────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │   Grafana    │
            │  Dashboard   │
            └──────────────┘
```

## Project Structure

```
MLOps_Final_Project/
├── . github/
│   └── workflows/
│       └── CICD.yml          # CI/CD pipeline configuration
├── api/
│   ├── api.py                # FastAPI application
│   └── metrics_recorder.py   # Prometheus metrics
├── data/                     # Data storage (DVC tracked)
├── logic/
│   └── regressor. py          # Model inference logic
├── notebooks/                # Jupyter notebooks for exploration
├── production/               # Production model artifacts (ONNX)
├── report/                   # Analysis reports and visualizations
├── scripts/
│   ├── train.py             # Model training script
│   └── serialize.py         # Model serialization to ONNX
├── templates/               # HTML templates for web UI
├── tests/                   # Test suite
├── third_party/            # External dependencies
├── utils/                  # Utility functions
├── Dockerfile              # Main application container
├── Dockerfile. prometheus   # Prometheus container
├── Makefile               # Development commands
├── pyproject.toml         # Project dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized deployment)
- DVC (for data versioning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Iker-Jauregui/MLOps_Final_Project. git
   cd MLOps_Final_Project
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. **Pull data with DVC**
   ```bash
   dvc pull
   ```

### Running the API

**Local Development:**
```bash
# Using the Makefile
make run

# Or directly with uvicorn
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

**Using Docker:**
```bash
# Build the image
docker build -t mlops-music-revenue .

# Run the container
docker run -p 8000:8000 mlops-music-revenue
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## Development

### Training Models

Train a Random Forest model with hyperparameter optimization: 

```bash
# Make sure training dependencies are installed
uv sync

# Run training with MLflow tracking
python scripts/train.py
```

The training script will: 
- Load and preprocess data
- Perform hyperparameter optimization with Optuna
- Log experiments to MLflow
- Save the best model
- Generate SHAP explanations

### Model Serialization

Convert trained models to ONNX format:

```bash
python scripts/serialize.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code with Black
black .

# Lint code with Pylint
pylint api/ logic/ scripts/

# Or use the Makefile
make lint
```

## Deployment

### Docker

**Build and run the main application:**
```bash
docker build -t mlops-music-revenue .
docker run -d -p 8000:8000 --name music-api mlops-music-revenue
```

**Run with Prometheus monitoring:**
```bash
# Build Prometheus container
docker build -f Dockerfile.prometheus -t mlops-prometheus .

# Run Prometheus
docker run -d -p 9090:9090 --name prometheus mlops-prometheus
```

### Monitoring

The API exposes Prometheus metrics at `/metrics`:

- **Request count**: Total number of predictions
- **Request latency**:  Prediction response times
- **Model predictions**: Distribution of predicted values
- **Prediction distribution**: Histogram of predictions

**Grafana Dashboard:**
- Access Prometheus at `http://localhost:9090`
- Configure Grafana to visualize metrics with custom dashboards
- Monitor API performance, prediction distributions, and system health

## API Documentation

### Endpoints

#### Make Prediction
```
POST /predict
```
Predict revenue for music track features.

**Request Body:**
```json
{
  "feature1": 0.5,
  "feature2": 1.2,
  ... 
}
```

#### Web UI
```
GET /
```
Interactive web interface for making predictions. 

#### Metrics
```
GET /metrics
```
Prometheus metrics endpoint.

Visit `/docs` for complete API documentation with interactive examples.

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

### Pipeline Stages

1. **Code Quality Checks**
   - Linting with Pylint
   - Code formatting with Black
   - Type checking

2. **Testing**
   - Unit tests with pytest
   - Integration tests
   - Coverage reporting

3. **Build**
   - Docker image creation
   - Multi-stage builds for optimization

4. **Deployment** (on main branch)
   - Automated deployment to production
   - Model artifact management

The pipeline runs on every push and pull request. Check the [CICD badge](#-mlops-final-project-music-track-revenue-prediction) for current status.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Iker Jáuregui** - [Iker-Jauregui](https://github.com/Iker-Jauregui)
- **Melchor Lafuente** - [NeoLafuente](https://github.com/NeoLafuente)
- **Iker Urdiroz** - [ikerua](https://github.com/ikerua)

---

**Made with <3 for the MLOps Final Project**
