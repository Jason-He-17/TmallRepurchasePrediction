import pandas as pd
import numpy as np

def drop_high_corr_features(df, label_col='label', threshold=0.85):
    # Only check numeric columns
    numeric_cols = df.select_dtypes(include=['int','float','uint','double']).columns.tolist()
    # Remove the label column and non-feature columns
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if 'origin' in numeric_cols:
        numeric_cols.remove('origin')

    corr_matrix = df[numeric_cols].corr().abs()
    # Set upper triangle to NaN to avoid duplicate comparisons
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns that have a correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Exclude these highly correlated columns
    remaining_cols = [c for c in numeric_cols if c not in to_drop]

    return to_drop, remaining_cols