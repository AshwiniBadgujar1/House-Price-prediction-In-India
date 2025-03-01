import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Streamlit App Title
st.title("House Price Prediction in India")

# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Data Preprocessing
    categorical_columns = ["condition of the house", "grade of the house", "waterfront present"]
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le 
    
    df = df.dropna()  # Handling missing values
    
    # Feature Selection Options
    all_features = [
        "number of bedrooms", "number of bathrooms", "living area", "lot area",
        "number of floors", "waterfront present", "number of views", 
        "condition of the house", "grade of the house", 
        "Area of the house(excluding basement)", "Area of the basement",
        "Built Year", "Renovation Year", "Lattitude", "Longitude"
    ]
    
    selected_features = st.multiselect("Select Features for Prediction", all_features, default=all_features)
    target = "Price"
    
    if all(col in df.columns for col in selected_features) and target in df.columns:
        X = df[selected_features]
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate Errors
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        st.write("### Model Evaluation")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Root Mean Squared Error: {rmse:.2f}")
        
        # Visualization
        st.write("### Actual vs Predicted House Prices")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual vs Predicted House Prices in India (Random Forest)")
        st.pyplot(fig)
    else:
        st.error("The dataset does not contain all required columns. Please check and upload a correct file.")
