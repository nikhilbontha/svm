
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Electric Car Price Prediction using SVR")


    # Load Dataset
    df = pd.read_csv("ElectricCarData_Clean (1).csv")

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Information")
    st.write(df.info())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Features and Target
    x = df.drop('PriceEuro', axis=1)
    y = df['PriceEuro']

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Convert categorical data into numerical data
    le = LabelEncoder()
    x_train = x_train.apply(le.fit_transform)
    x_test = x_test.apply(le.fit_transform)

    # Create SVR Model
    svr_classifier = SVR(kernel='rbf', C=1.0, gamma='scale')

    # Train Model
    svr_classifier.fit(x_train, y_train)

    # Predictions
    y_pred = svr_classifier.predict(x_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R2 Score: {r2}")

    # Show Predictions
    results = pd.DataFrame({
        'Actual Price': y_test.values,
        'Predicted Price': y_pred
    })

    st.subheader("Prediction Results")
    st.write(results.head())

else:
    st.info("Please upload a CSV file to continue.")
