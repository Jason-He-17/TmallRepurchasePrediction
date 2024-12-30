import pandas as pd
import numpy as np

def drop_high_corr_features(df, label_col='label', threshold=0.85):
    # 只对数值列做检查
    numeric_cols = df.select_dtypes(include=['int','float','uint','double']).columns.tolist()
    # 移除标签列与非特征列
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if 'origin' in numeric_cols:
        numeric_cols.remove('origin')

    corr_matrix = df[numeric_cols].corr().abs()
    # 上三角置空，避免重复比较
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 找到大于阈值的列
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # 剔除掉这些高度相关的列
    remaining_cols = [c for c in numeric_cols if c not in to_drop]

    return to_drop, remaining_cols