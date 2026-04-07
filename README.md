# xgboost-app

Practical task from machine learning.

XGBoost (ucenie suborom metod na vahovanych prikladoch).

## Overview

This project is an interactive Streamlit application for training and evaluating an `XGBClassifier` on built-in scikit-learn datasets.

Available datasets:
- Iris
- Wine
- Breast Cancer
- Digits

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

## Theory

### 1. The problem with weak classifiers

A **weak classifier** is a model that performs only slightly better than random guessing — low Precision, Recall, F1, and Accuracy.
This is often caused by a small or low-quality training set, or by using a model that is too simple for the problem.
Ensemble methods address this by combining many weak classifiers into one strong classifier through voting (classification) or averaging (regression).
The key insight: a group of imperfect models that each make *different* mistakes can collectively outperform any single model.

---

### 2. Boosting — the core idea

Boosting trains classifiers **sequentially**. Each new model focuses on the examples the previous one got wrong
by increasing their weights in the training distribution. This is called **error-driven learning**.

The final prediction is a weighted vote across all M weak classifiers:

$$H(d, c) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m \cdot H_m(d, c)\right)$$

where $H_m$ is the m-th weak classifier and $\alpha_m$ is its weight — proportional to its accuracy.
More accurate classifiers get a louder vote. This is the AdaBoost.MH formulation from Schapire & Singer (1999).

Unlike **bagging** (which trains classifiers in parallel on random subsets), boosting is **sequential** —
each model depends on the errors of the previous one, making it more powerful but also more sensitive to noisy data.

---

### 3. From Boosting to Gradient Boosting

AdaBoost reweights training samples to focus on misclassified examples.
**Gradient Boosting** generalizes this idea: instead of reweighting samples, each new tree is trained
to predict the **residual errors** (gradients of the loss function) of the current ensemble.

If the current ensemble predicts 0.7 for a sample whose true value is 1.0,
the next tree learns to predict the residual 0.3. The ensemble improves by adding corrective trees.
The learning rate $\eta$ controls how much each tree contributes:

$$\hat{y}^{(m)} = \hat{y}^{(m-1)} + \eta \cdot h_m(x)$$

A smaller $\eta$ requires more trees but generalizes better.

---

### 4. XGBoost — what makes it different

| Feature | Description |
|---|---|
| **Regularization (L1 + L2)** | Standard Gradient Boosting has no penalty on tree weights. XGBoost adds L1 and L2 regularization directly into the objective, reducing overfitting — especially on small datasets. |
| **Second-order gradients** | Standard GBM uses only the first derivative of the loss. XGBoost also uses the second derivative (Hessian), giving a more accurate approximation of the loss surface and faster convergence. |
| **Column subsampling** | Like Random Forests, XGBoost randomly selects a subset of features per tree. This de-correlates the trees and reduces variance. |
| **Parallel tree construction** | Trees are built using a parallelized split-finding algorithm, making XGBoost significantly faster than naive GBM implementations despite the sequential nature of boosting. |

---

### 5. Key hyperparameters

| Parameter | Too low | Too high | Typical range |
|---|---|---|---|
| `n_estimators` | Underfitting — ensemble too weak | Overfitting + slow training | 100 – 500 |
| `max_depth` | Underfitting — trees too shallow | Overfitting — trees memorize noise | 3 – 6 |
| `learning_rate` (η) | Slow convergence, needs many trees | Overfitting — each tree overcorrects | 0.01 – 0.3 |

A practical rule: **lower learning rate + more estimators** generally outperforms **high learning rate + fewer estimators**, at the cost of training time.

---

### 6. When to use XGBoost

XGBoost excels on **structured/tabular data** with numerical and categorical features,
especially for medium-sized datasets (thousands to hundreds of thousands of samples).
It is a strong default choice for most classification and regression tasks.

It is less suitable for **image, audio, or raw text** where deep learning has a structural advantage.
It can also struggle when training data is very small — boosting risks chasing noise,
as visible in the Breast Cancer results on the Compare tab.

---

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
