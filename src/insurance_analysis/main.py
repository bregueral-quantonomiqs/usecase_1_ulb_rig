"""
Main module with example usage of the insurance analysis package.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from insurance_analysis.data_loader import load_csv_data, preprocess_insurance_data
from insurance_analysis.analysis import get_data_summary, train_claim_model
from insurance_analysis.visualization import (
    plot_claim_distribution,
    plot_correlation_matrix,
    plot_feature_importance
)


def main():
    """
    Example workflow using the insurance analysis package.
    """
    # Get the root project directory
    project_dir = Path(__file__).resolve().parent.parent.parent.parent
    
    # Load data
    data_path = project_dir / "data" / "mcc.csv"
    print(f"Loading data from: {data_path}")
    
    try:
        df = load_csv_data(data_path)
        print(f"Successfully loaded data with shape: {df.shape}")
        
        # Preprocess data
        processed_df = preprocess_insurance_data(df)
        print("Data preprocessing completed")
        
        # Get data summary
        summary = get_data_summary(processed_df)
        print("\nData Summary:")
        print(f"Shape: {summary['shape']}")
        print(f"Columns: {summary['columns']}")
        print("Missing values:")
        for col, count in summary['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}")
        
        # Create output directory for visualizations
        output_dir = project_dir / "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Example visualization (this will depend on the actual data)
        # For demonstration purposes, let's assume we have some numeric columns
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 3:  # Need at least some numeric columns
            # Let's assume the first numeric column is our target
            target_col = numeric_cols[0]
            feature_cols = numeric_cols[1:]
            
            # Train a model
            model, metrics, X_test, y_test = train_claim_model(
                processed_df, target_col, feature_cols
            )
            
            print("\nModel Training Results:")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")
            
            # Plot feature importance
            fig = plot_feature_importance(
                feature_cols, 
                metrics['coefficients'].values(),
                save_path=output_dir / "feature_importance.png"
            )
            plt.close(fig)
            
            # Plot correlation matrix
            fig = plot_correlation_matrix(
                processed_df[numeric_cols],
                save_path=output_dir / "correlation_matrix.png"
            )
            plt.close(fig)
            
            print("\nVisualizations saved to:", output_dir)
            
    except Exception as e:
        print(f"Error in analysis: {e}")


if __name__ == "__main__":
    main()