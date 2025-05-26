# Stroke Risk Predictor

## Table of Contents

- [Overview](#overview)
- [Interface](#interface)
- [Features](#features)
- [Model Details](#model-details)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Using uv (Recommended)](#using-uv-recommended)
  - [Using pip (Alternative)](#using-pip-alternative)
- [Development Setup](#development-setup)
  - [Environment Configuration](#environment-configuration)
- [Local Execution](#local-execution)
- [Deployment](#deployment)
- [Testing](#testing)
- [License](#license)

## Overview

Machine learning-based web application designed to assess stroke risk based on health and lifestyle factors. The system processes patient data through a CatBoost model to provide risk assessments, helping healthcare professionals identify potential stroke risks early for timely intervention.

## Interface

![Web App Interface](images/web_app.png)

## Features

- Comprehensive health data analysis
- Advanced feature engineering implementation
- Multiple model evaluation framework
- High-recall optimization
- Flask-based web interface
- Google Cloud Platform deployment
- Automated testing suite
- Containerized deployment

## Model Details

Current implemented models evaluated:
1. Logistic Regression
2. XGBoost
3. CatBoost (selected as final model)

## Prerequisites

- Python 3.9+ (check `.python-version` file for the current required version)
- Docker
- Google Cloud SDK
- Flask
- scikit-learn
- CatBoost

## Installation

### Using uv (Recommended)

1. Install uv:
   ```bash
   # On Unix/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/vytautas-bunevicius/stroke-risk-predictor.git
   cd stroke-risk-predictor
   ```

3. Create and activate virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

### Using pip (Alternative)

1. Clone the repository:
   ```bash
   git clone https://github.com/vytautas-bunevicius/stroke-risk-predictor.git
   cd stroke-risk-predictor
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Setup

### Environment Configuration

1. Create `.env` file in project root:
   ```env
   FLASK_ENV=development
   MODEL_PATH=models/catboost_model.pkl
   PORT=5000
   ```

2. Configure Google Cloud services:
   - App Engine
   - Cloud Storage (for model storage)
   - Secret Manager

## Local Execution

Run the application locally:
```bash
python src/stroke_risk_predictor/app.py
```

Visit `http://localhost:5000` in your browser.

## Deployment

The application is deployed on Google Cloud Platform App Engine:

1. Configure deployment:
   ```bash
   gcloud config set project your-project-id
   ```

2. Deploy:
   ```bash
   gcloud app deploy
   ```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## License

This project is released under the [Unlicense](https://unlicense.org/). This means you can copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

See the [UNLICENSE](UNLICENSE) file for more details.