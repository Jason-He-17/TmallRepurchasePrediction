# helper.py

import numpy as np
import pandas as pd

def _cap_values(series, upper_percentile=99):
    """
    Cap the feature values at the given percentile to mitigate the negative impact of extreme values on the model.
    """
    cap = np.percentile(series.dropna(), upper_percentile)
    return np.where(series > cap, cap, series)

def _bin_values(series, bins=5):
    """
    Convert continuous numeric features into binned categorical features using equal-frequency binning (qcut).
    """
    try:
        return pd.qcut(series, q=bins, labels=False, duplicates='drop')
    except ValueError:
        # If binning fails (e.g., all values are the same), return a series of 0s
        return pd.Series([0] * len(series), index=series.index)

def remove_original_columns_if_binned(df):
    """
    For each *_bin column, if the binned column has more than one unique value,
    it is considered that binning was successful, and the original column with the same name (excluding the _bin suffix) is automatically removed.
    """
    # First, collect all columns ending with `_bin`
    binned_cols = [c for c in df.columns if c.endswith('_bin')]
    
    for bin_col in binned_cols:
        # If the binned column actually has multiple bins (unique values)
        if df[bin_col].nunique() > 1:
            # Construct the original feature column name
            orig_col = bin_col[:-4]  # Remove "_bin"
            # If the original column exists, drop it
            if orig_col in df.columns:
                df.drop(columns=[orig_col], inplace=True)
                
    return df