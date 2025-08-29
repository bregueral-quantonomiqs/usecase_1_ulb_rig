"""
Data loading and preprocessing module.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_csv_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str or Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def preprocess_insurance_data(df):
    """
    Preprocess insurance data.
    
    Args:
        df (pd.DataFrame): Raw insurance data.
        
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Example preprocessing steps
    # Fill missing values
    processed_df = processed_df.fillna(0)
    
    # Convert date columns to datetime if they exist
    date_cols = [col for col in processed_df.columns if 'date' in col.lower()]
    for col in date_cols:
        try:
            processed_df[col] = pd.to_datetime(processed_df[col])
        except:
            pass
    
    return processed_df