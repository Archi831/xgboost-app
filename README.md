# xgboost-app

Practical task from machine learning.

XGBoost (ucenie suborom metod na vahovanych prikladoch).

## Overview

This project is an interactive Streamlit application for training and evaluating an `XGBClassifier` on built-in scikit-learn datasets.

Available datasets:
- Iris
- Wine
- Breast Cancer

Main features:
- Dataset selection from sidebar
- Hyperparameter tuning (`n_estimators`, `max_depth`, `learning_rate`)
- Adjustable train/test split
- Model training with one click
- Evaluation metrics: Accuracy, Precision, Recall, F1
- Visual analysis: Feature Importance and Confusion Matrix

## Tech Stack

- Python
- Streamlit
- XGBoost
- scikit-learn
- pandas
- plotly

## Project Structure

- `app.py` - main Streamlit app
- `requirements.txt` - pinned dependencies

## Setup

1. Open terminal in the project folder:

```powershell
cd xgboost-app
```

2. Create a virtual environment:

```powershell
python -m venv venv
```

3. Activate the virtual environment (Windows PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

Start the Streamlit app:

```powershell
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Notes

- The app currently focuses on training and analysis workflows.
- Tabs `Theory`, `Predict`, and `Compare` are present in UI and can be expanded with additional content.
