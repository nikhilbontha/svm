import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


st.title("Support Vector Machine (SVM) Classification")

st.header("1. Load Iris Dataset")

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

st.dataframe(df.head())

st.header("2. Configure SVM Model")

kernel = st.selectbox(
    "Select Kernel",
    ["linear", "rbf", "poly", "sigmoid"]
)

c_value = st.slider(
    "Select C Value",
    min_value=0.1,
    max_value=10.0,
    value=1.0
)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel=kernel, C=c_value)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.write(f"Accuracy: {accuracy:.4f}")

st.subheader("Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    columns=iris.target_names,
    index=iris.target_names
)

st.dataframe(cm_df)

st.success("SVM Classification Completed Successfully!")
