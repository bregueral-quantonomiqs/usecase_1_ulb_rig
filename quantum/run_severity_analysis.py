#!/usr/bin/env python
"""
Simplified severity analysis script for insurance claims.
This script focuses only on the severity aspect of the enhanced quantum solution.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.ensemble import GradientBoostingRegressor

# Visualization
from quantum.visualization import (
    visualize_regression_results,
    visualize_residuals,
    visualize_residual_histogram,
    visualize_residuals_qq,
    visualize_learning_curve,
)

# Constants
RANDOM_STATE = 42
NUMERIC = ["OwnersAge", "VehicleAge", "BonusClass"]
CATEG = ["Gender", "Zone", "Class"]  # treat Zone & Class as categories even if int-coded

def load_clean(path):
    """Load and clean the MCC dataset with focus on severity."""
    df = pd.read_csv(path)
    
    # Keep positive exposure
    df = df.loc[df["Duration"] > 0].copy()
    
    # Normalize gender labels (Swedish dataset uses 'M'/'K')
    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"M": "M", "K": "F"}).fillna(df["Gender"])
    
    # Ensure expected dtypes
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    for c in ["Zone", "Class"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    
    # Targets
    df["exposure"] = df["Duration"]
    df["claims"] = df["NumberClaims"].astype(int)
    df["total_cost"] = df["ClaimCost"].astype(float)
    
    # Severity only for claims > 0
    df["severity"] = np.where(df["claims"] > 0, df["total_cost"] / df["claims"], np.nan)
    
    # Add engineered features
    df = add_engineered_features(df)
    
    return df

def add_engineered_features(df):
    """Add domain-specific engineered features for better prediction."""
    df_enhanced = df.copy()
    
    # Add interaction terms
    df_enhanced["age_class"] = df_enhanced["OwnersAge"] * df_enhanced["Class"]
    df_enhanced["age_zone"] = df_enhanced["OwnersAge"] * df_enhanced["Zone"]
    
    # Add exposure-related features
    df_enhanced["log_exposure"] = np.log1p(df_enhanced["exposure"])
    df_enhanced["exposure_bonus"] = df_enhanced["log_exposure"] * df_enhanced["BonusClass"]
    
    # Add non-linear transformations
    df_enhanced["age_squared"] = df_enhanced["OwnersAge"] ** 2
    df_enhanced["log_vehicle_age"] = np.log1p(df_enhanced["VehicleAge"])
    
    # Add risk score
    df_enhanced["risk_score"] = (
        df_enhanced["OwnersAge"] / 100 +
        df_enhanced["VehicleAge"] / 5 +
        df_enhanced["Zone"] / df_enhanced["Zone"].max() +
        (5 - df_enhanced["BonusClass"]) / 5
    )
    
    return df_enhanced

def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

def run_severity_analysis(data_path="data/mcc.csv", results_dir="results"):
    """Run severity analysis on insurance claims data."""
    print(f"Loading data from {data_path}...")
    df = load_clean(data_path)
    
    # Filter to claims > 0 for severity analysis
    df_sev = df.loc[df["claims"] > 0].copy()
    
    # Define features
    engineered_features = [
        "age_class", "age_zone", "log_exposure", "exposure_bonus", 
        "age_squared", "log_vehicle_age", "risk_score"
    ]
    
    # Use all available features
    available_features = [f for f in engineered_features if f in df_sev.columns]
    all_features = NUMERIC + CATEG + available_features
    
    # Prepare data
    X = df_sev[all_features]
    y = df_sev["severity"].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC + available_features),
            ('cat', categorical_transformer, CATEG)
        ])
    
    # Create and train models
    print("Training classical KernelRidge with target log-transform + tuning...")
    base = KernelRidge(kernel='rbf')
    ttr = TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)

    # Tune alpha & gamma (kept small grid for runtime)
    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0],
        'regressor__gamma': [0.01, 0.1, 1.0]
    }
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GridSearchCV(ttr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1))
    ])

    model.fit(X_train, y_train)
    
    # Make predictions
    # Extract best estimator if GridSearch used
    try:
        inner = model.named_steps['regressor']
        if hasattr(inner, 'best_params_'):
            print(f"Best params: {inner.best_params_}")
    except Exception:
        pass

    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_regression(y_test, y_pred)
    
    # Print results
    print("\nClassical Kernel Ridge Regression Results (log-transformed target + tuned):")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R²: {metrics['r2']:.2f}")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Train GradientBoosting baseline (on preprocessed features)
    print("Training GradientBoosting baseline for severity...")
    try:
        # Transform features explicitly to reuse the fitted preprocessor
        X_train_trans = model.named_steps['preprocessor'].transform(X_train)
        X_test_trans = model.named_steps['preprocessor'].transform(X_test)
        try:
            gbr = HGBR()
            param_grid_gb = {
                'learning_rate': [0.05, 0.1],
                'max_depth': [None, 6, 10],
                'max_iter': [200, 400]
            }
        except Exception:
            gbr = GradientBoostingRegressor()
            param_grid_gb = {
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'n_estimators': [200, 400]
            }
        gb_cv = GridSearchCV(gbr, param_grid=param_grid_gb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gb_cv.fit(X_train_trans, y_train)
        y_pred_gb = gb_cv.predict(X_test_trans)
        print(f"Best GB params (severity): {gb_cv.best_params_}")
        metrics_gb = evaluate_regression(y_test, y_pred_gb)
    except Exception as e:
        print(f"Warning: GB training failed: {e}")
        y_pred_gb = np.zeros_like(y_test)
        metrics_gb = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 0.0}

    # Save results to file
    results_file = os.path.join(results_dir, "severity_analysis_results.txt")
    with open(results_file, 'w') as f:
        f.write("Severity Analysis Results\n")
        f.write("========================\n\n")
        f.write(f"Data source: {data_path}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n\n")
        f.write("Classical Kernel Ridge Regression Metrics (log-transformed target + tuned):\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write("\nGradientBoosting Regression Metrics:\n")
        f.write(f"MAE: {metrics_gb['mae']:.4f}\n")
        f.write(f"MSE: {metrics_gb['mse']:.4f}\n")
        f.write(f"RMSE: {metrics_gb['rmse']:.4f}\n")
        f.write(f"R²: {metrics_gb['r2']:.4f}\n")
    
    # Save diagnostic plots
    try:
        visualize_regression_results(
            y_test, y_pred,
            title="Severity Actual vs Predicted",
            filename="severity_actual_vs_predicted.png",
            results_dir=results_dir
        )
        visualize_learning_curve(
            model, X_train, y_train,
            title="Severity KRR Learning Curve (neg MSE)",
            filename="severity_krr_learning_curve.png",
            results_dir=results_dir,
            cv=3,
            scoring='neg_mean_squared_error'
        )
        visualize_residuals(
            y_test, y_pred,
            title="Severity Residuals vs Predicted",
            filename="severity_residuals_vs_predicted.png",
            results_dir=results_dir
        )
        visualize_residual_histogram(
            y_test, y_pred,
            title="Severity Residuals Histogram",
            filename="severity_residuals_histogram.png",
            results_dir=results_dir
        )
        visualize_residuals_qq(
            y_test, y_pred,
            title="Severity Residuals Q-Q Plot",
            filename="severity_residuals_qq.png",
            results_dir=results_dir
        )
        # GB visuals
        visualize_regression_results(
            y_test, y_pred_gb,
            title="GB Severity Actual vs Predicted",
            filename="severity_gb_actual_vs_predicted.png",
            results_dir=results_dir
        )
        visualize_residuals(
            y_test, y_pred_gb,
            title="GB Severity Residuals vs Predicted",
            filename="severity_gb_residuals_vs_predicted.png",
            results_dir=results_dir
        )
        visualize_residual_histogram(
            y_test, y_pred_gb,
            title="GB Severity Residuals Histogram",
            filename="severity_gb_residuals_histogram.png",
            results_dir=results_dir
        )
        visualize_residuals_qq(
            y_test, y_pred_gb,
            title="GB Severity Residuals Q-Q Plot",
            filename="severity_gb_residuals_qq.png",
            results_dir=results_dir
        )
        visualize_learning_curve(
            gb_cv.best_estimator_ if 'gb_cv' in locals() else gbr,
            X_train_trans if 'X_train_trans' in locals() else X_train,
            y_train,
            title="Severity GB Learning Curve (neg MSE)",
            filename="severity_gb_learning_curve.png",
            results_dir=results_dir,
            cv=3,
            scoring='neg_mean_squared_error'
        )
    except Exception as e:
        print(f"Warning: Could not save diagnostic plots: {e}")

    print(f"\nResults saved to {results_file}")
    return metrics

if __name__ == "__main__":
    run_severity_analysis()
