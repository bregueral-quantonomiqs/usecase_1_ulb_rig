"""
Visualization module for insurance data analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_claim_distribution(df, claim_col, figsize=(10, 6), save_path=None):
    """
    Plot the distribution of insurance claims.
    
    Args:
        df (pd.DataFrame): Insurance data.
        claim_col (str): Name of the claim amount column.
        figsize (tuple): Figure size (width, height).
        save_path (str, optional): Path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    plt.figure(figsize=figsize)
    
    # Create histogram with KDE
    sns.histplot(df[claim_col], kde=True)
    
    plt.title(f'Distribution of {claim_col}')
    plt.xlabel(claim_col)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_correlation_matrix(df, figsize=(12, 10), save_path=None):
    """
    Plot correlation matrix for numeric columns in the data.
    
    Args:
        df (pd.DataFrame): Insurance data.
        figsize (tuple): Figure size (width, height).
        save_path (str, optional): Path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Compute correlation matrix
    corr = numeric_df.corr()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_feature_importance(feature_names, coefficients, figsize=(10, 8), save_path=None):
    """
    Plot feature importance based on model coefficients.
    
    Args:
        feature_names (list): Names of the features.
        coefficients (list or array): Model coefficients.
        figsize (tuple): Figure size (width, height).
        save_path (str, optional): Path to save the figure.
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefficients)
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=figsize)
    
    # Create barplot
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    
    plt.title('Feature Importance')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()