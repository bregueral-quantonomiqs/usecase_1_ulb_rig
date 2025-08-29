"""
Tests for the data_loader module.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from insurance_analysis.data_loader import load_csv_data, preprocess_insurance_data


def test_preprocess_insurance_data():
    """Test the preprocessing function."""
    # Create a test dataframe
    test_data = {
        "id": [1, 2, 3, 4, 5],
        "amount": [100.0, 200.0, None, 400.0, 500.0],
        "policy_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
    }
    df = pd.DataFrame(test_data)
    
    # Process the dataframe
    processed_df = preprocess_insurance_data(df)
    
    # Check that missing values are filled
    assert processed_df["amount"].isna().sum() == 0
    
    # Check that policy_date is converted to datetime
    assert pd.api.types.is_datetime64_dtype(processed_df["policy_date"])
    
    # Original dataframe should be unchanged
    assert df["amount"].isna().sum() == 1
    assert not pd.api.types.is_datetime64_dtype(df["policy_date"])