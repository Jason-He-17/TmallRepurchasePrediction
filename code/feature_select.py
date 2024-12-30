# feature_select.py
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(data, label_col='label', k=50):
    # 只针对数值型特征做筛选，去除非数值列、标签列、origin列
    numeric_cols = data.select_dtypes(include=['int64','float64','int32','float32','uint32','uint16']).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if 'origin' in numeric_cols:
        numeric_cols.remove('origin')

    X = data[numeric_cols]
    y = data[label_col]

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    selected_cols = [col for col, m in zip(numeric_cols, mask) if m]
    return selected_cols