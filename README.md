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

Tabs:
- **Theory** — ensemble learning background, AdaBoost formulation, Gradient Boosting, XGBoost specifics, hyperparameter guide
- **Training** — dataset preview, evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- **Analysis** — Feature Importance bar chart, Confusion Matrix, Performance vs `n_estimators` line chart
- **Predict** — per-feature number inputs (bounded by dataset min/max, defaulting to mean), predicted class label and probability bar chart
- **Compare** — side-by-side Accuracy and F1 table for XGBoost, Decision Tree, and Bagging; Accuracy/F1 vs `n_estimators` line chart for XGBoost vs Bagging

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

- Switching datasets resets the trained model — retrain after changing the dataset selection.
- The Performance vs n_estimators and Compare charts iterate `n_estimators` from 10 to 300 in steps of 20; results are cached via `@st.cache_data` so re-renders are instant.
- All metrics use macro averaging to handle multi-class datasets uniformly.
