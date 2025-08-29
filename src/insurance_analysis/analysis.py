"""
Analysis module for insurance data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def get_data_summary(df):
    """
    Generate a summary of the data.
    
    Args:
        df (pd.DataFrame): Insurance data.
        
    Returns:
        dict: Summary statistics.
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict()
    }
    return summary


def train_claim_model(df, target_col, feature_cols, test_size=0.2, random_state=42):
    """
    Train a simple linear regression model to predict insurance claims.
    
    Args:
        df (pd.DataFrame): Insurance data.
        target_col (str): Name of the target column.
        feature_cols (list): List of feature column names.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (model, test_metrics, X_test, y_test)
    """
    # Extract features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "coefficients": dict(zip(feature_cols, model.coef_)),
        "intercept": model.intercept_
    }
    
    return model, metrics, X_test, y_test