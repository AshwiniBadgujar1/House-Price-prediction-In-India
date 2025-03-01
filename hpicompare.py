import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
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
        
        # Train Models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[name] = (mae, rmse)
        
        # Display Results
        st.write("### Model Evaluation Comparison")
        results_df = pd.DataFrame(results, index=["Mean Absolute Error", "Root Mean Squared Error"])
        st.write(results_df)
        
        # Visualization of Model Performance
        st.write("### Model Error Comparison")
        fig, ax = plt.subplots()
        results_df.T.plot(kind='bar', ax=ax)
        ax.set_ylabel("Error Value")
        ax.set_title("Comparison of Model Errors")
        st.pyplot(fig)
        
        # Model Selection Visualization
        st.write("### Best Model Selection")
        fig, ax = plt.subplots()
        rmse_values = results_df.loc["Root Mean Squared Error"]
        ax.plot(rmse_values.index, rmse_values.values, marker='o', linestyle='-', color='b')
        ax.set_ylabel("RMSE Value")
        ax.set_title("Model Performance Comparison")
        ax.grid(True)
        st.pyplot(fig)
        
        # Feature Importance (for Random Forest)
        rf_model = models["Random Forest"]
        feature_importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({"Feature": selected_features, "Importance": feature_importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        
        st.write("### Feature Importance (Random Forest)")
        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"], color='blue')
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance in House Price Prediction")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
        
        # Scatter Plot - Actual vs Predicted (Random Forest)
        st.write("### Actual vs Predicted House Prices (Random Forest)")
        fig, ax = plt.subplots()
        y_pred_rf = rf_model.predict(X_test)
        ax.scatter(y_test, y_pred_rf, alpha=0.5)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual vs Predicted House Prices in India (Random Forest)")
        st.pyplot(fig)
    else:
        st.error("The dataset does not contain all required columns. Please check and upload a correct file.")
