import streamlit as st

import pandas as pd

from xgboost import XGBClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import plotly.express as px 
import plotly.figure_factory as ff

@st.cache_data
def load_dataset(name):
    """Loads dataset and returns a formatted DataFrame."""
    # Map the selectbox strings directly to the sklearn functions
    loaders = {
        "Iris": load_iris,
        "Wine": load_wine,
        "Breast Cancer": load_breast_cancer
    }
    
    # Execute the mapped function
    dataset = loaders[name]()
    
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target
    return (df, dataset)

# --- Global Sidebar ---

selected_dataset = st.sidebar.selectbox(
    "Select dataset", 
    ["Iris", "Wine", "Breast Cancer"]
)

if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = selected_dataset

if st.session_state.last_dataset != selected_dataset:
    st.session_state.last_dataset = selected_dataset
    if "bst" in st.session_state:
        del st.session_state.bst

df, dataset = load_dataset(selected_dataset)

n_estimators  = st.sidebar.slider("n_estimators", min_value=10, max_value=300, value=100)
max_depth     = st.sidebar.slider("max_depth", min_value=1, max_value=10, value=3)
learning_rate = st.sidebar.slider("learning_rate", min_value=0.01, max_value=1.0, value=0.1)
tt_split = st.sidebar.slider("Train/Test Split (%)", min_value=10, max_value=90, value=80)

# --- Model training ---

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - tt_split) / 100, random_state=42
)

if st.sidebar.button("Train Model"):
    st.session_state.bst = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric="mlogloss")
    st.session_state.bst.fit(X_train, y_train)
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test    
    st.success("Model trained successfully!")

# --- Layout ---

theory, training, analysis, predict, compare = st.tabs(["Theory", "Training", "Analysis", "Predict", "Compare"])

with training:
    st.subheader(f"{selected_dataset} Dataset")
    
    # df.shape[1] includes the target column, so we subtract 1 for the feature count
    n_samples = df.shape[0]
    n_features = df.shape[1] - 1 
    
    st.write(f"{n_samples} samples, {n_features} features")
    st.dataframe(df)

    st.subheader("Metrics")
    if 'bst' in st.session_state:
        with st.spinner("Calculating metrics..."):
            acc = st.session_state.bst.score(st.session_state.X_test, st.session_state.y_test)
            y_pred = st.session_state.bst.predict(st.session_state.X_test)
            precision = precision_score(st.session_state.y_test, y_pred, average='macro')
            recall = recall_score(st.session_state.y_test, y_pred, average='macro')
            f1 = f1_score(st.session_state.y_test, y_pred, average='macro')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")
    else:
        st.warning("Train the model to see metrics.")


with analysis:
    st.subheader("Feature Importance")
    if 'bst' in st.session_state:
        importance = st.session_state.bst.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(importance_df, x="Feature", y="Importance", title="")
        st.plotly_chart(fig)
    else:
        st.warning("Train the model to see feature importance.")

    st.subheader("Confusion Matrix")
    if 'bst' in st.session_state:
        y_pred = st.session_state.bst.predict(st.session_state.X_test)
        cm = confusion_matrix(st.session_state.y_test, y_pred)

        fig = ff.create_annotated_heatmap(
            z=cm,
            x=list(dataset.target_names),
            y=list(dataset.target_names),
            colorscale="Blues",
            showscale=True
        )
        fig.update_layout(title="", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig)