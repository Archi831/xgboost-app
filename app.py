import streamlit as st

import pandas as pd

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
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
st.session_state.X_train = X_train
st.session_state.y_train = y_train
st.session_state.X_test = X_test
st.session_state.y_test = y_test

if st.sidebar.button("Train Model"):
    st.session_state.bst = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric="mlogloss")
    st.session_state.bst.fit(X_train, y_train)
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
        st.plotly_chart(fig, key="feature_importance")
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
        st.plotly_chart(fig, key="confusion")
    else:
        st.warning("Train the model to see the confusion matrix.")
    
    st.subheader("Performance vs n_estimators")
    @st.cache_data
    def train_with_estimators(max_depth, learning_rate, X_train, y_train, X_test, y_test, model_type="xgb"):
        values = {}
        for n in range(10, 210, 20):
            if model_type == "xgb":
                model = XGBClassifier(n_estimators=n, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric="mlogloss")
            else:
                model = BaggingClassifier(estimator=dt, n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            l = [model.score(X_test, y_test), f1_score(y_test, model.predict(X_test), average='macro')]
            values[str(n)] = l
        values_df = pd.DataFrame(values, index=["Accuracy", "F1-Score"]).T
        return values_df

    if "bst" in st.session_state:
        with st.spinner("Training with different n_estimators..."):
            values_df = train_with_estimators(
                max_depth, learning_rate,
                st.session_state.X_train,
                st.session_state.y_train,
                st.session_state.X_test,
                st.session_state.y_test,
                model_type="xgb"
            )
            fig = px.line(values_df, x=values_df.index, y=["Accuracy", "F1-Score"], markers=True)
            fig.update_layout(
                title="", 
                xaxis_title="n_estimators", 
                yaxis_title="Score", 
                )
            st.plotly_chart(fig, key="performance_n_estimators")
    else:
        st.warning("Train the model to see performance comparison.")
            

with compare:
    st.subheader("Compare with other models")
    if 'bst' in st.session_state:
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(st.session_state.X_train, st.session_state.y_train)
        dt_acc = dt.score(st.session_state.X_test, st.session_state.y_test)
        dt_f1 = f1_score(st.session_state.y_test, dt.predict(st.session_state.X_test), average='macro')

        bag = BaggingClassifier(estimator=dt, n_estimators=n_estimators, random_state=42)
        bag.fit(st.session_state.X_train, st.session_state.y_train)
        bag_acc = bag.score(st.session_state.X_test, st.session_state.y_test)
        bag_f1 = f1_score(st.session_state.y_test, bag.predict(st.session_state.X_test), average='macro')

        xgb_acc = st.session_state.bst.score(st.session_state.X_test, st.session_state.y_test)
        xgb_f1 = f1_score(st.session_state.y_test, st.session_state.bst.predict(st.session_state.X_test), average='macro')

        comparison_df = pd.DataFrame({
            "Accuracy": [xgb_acc, dt_acc, bag_acc],
            "F1-Score": [xgb_f1, dt_f1, bag_f1]
        })
        models_df = pd.DataFrame({
            "Model": ["XGBoost", "Decision Tree", "Bagging"]
        })
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(models_df, hide_index=True, use_container_width=True)
        with col2:
            st.dataframe(comparison_df.style.highlight_max(axis=0), hide_index=True, use_container_width=True)

        with st.spinner("Training with different estimators..."):
            xgb_values_df = train_with_estimators(
                max_depth, learning_rate,
                st.session_state.X_train,
                st.session_state.y_train,
                st.session_state.X_test,
                st.session_state.y_test,
                model_type="xgb"
            )
            bagging_values_df = train_with_estimators(
                max_depth, learning_rate,
                st.session_state.X_train,
                st.session_state.y_train,
                st.session_state.X_test,
                st.session_state.y_test,
                model_type="bagging"
            )
            values_df = pd.concat([xgb_values_df, bagging_values_df], keys=["XGBoost", "Bagging"], names=["Model", "n_estimators"])
            values_df = values_df.reset_index()
            fig = px.line(values_df, x="n_estimators", y=["Accuracy", "F1-Score"], markers=True, color="Model")
            fig.update_layout(
                title="", 
                xaxis_title="n_estimators", 
                yaxis_title="Score"
                )
            st.plotly_chart(fig, key="compare_estimators")
    else:
        st.warning("Train the model to see comparisons.")

with predict:
    st.subheader("Make a Prediction")
    st.write("Input feature values to predict the class label. Adjust the values for each feature based on the dataset's range.")
    st.write("*Default values are set to the mean of each feature in the training set.*")
    if 'bst' in st.session_state:
        input_data = {}
        grid = st.columns(3)
        for i, feature in enumerate(X.columns):
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())
            input_data[feature] = grid[i % 3].number_input(feature, min_value=min_val, max_value=max_val, value=mean_val)

        div = st.columns(2)
        if div[0].button("Predict"):
            input_df = pd.DataFrame([input_data])
            st.session_state.prediction = st.session_state.bst.predict(input_df)[0]
            st.session_state.predicted_class = dataset.target_names[st.session_state.prediction]
            div[1].success(f"{st.session_state.predicted_class}")
            st.session_state.probabilities = st.session_state.bst.predict_proba(pd.DataFrame([input_data]))[0]

        if "probabilities" in st.session_state and "predicted_class" in st.session_state:
            prob_df = pd.DataFrame({
                "Class": dataset.target_names,
                "Probability": st.session_state.probabilities
            })
            st.subheader("Prediction Probabilities")
            prob_df["Predicted"] = prob_df["Class"] == st.session_state.predicted_class
            fig = px.bar(prob_df, x="Class", y="Probability", color="Predicted")
            st.plotly_chart(fig)
    else:
        st.warning("Train the model to make predictions.")
                
