# helper.py

import numpy as np
import pandas as pd

def _cap_values(series, upper_percentile=99):
    """
    将特征在给定的百分位数上进行截断，避免极端值对模型产生负面影响。
    """
    cap = np.percentile(series.dropna(), upper_percentile)
    return np.where(series > cap, cap, series)

def _bin_values(series, bins=5):
    """
    使用等频分箱（qcut）将连续数值特征转换为分箱后的分类特征。
    """
    try:
        return pd.qcut(series, q=bins, labels=False, duplicates='drop')
    except ValueError:
        # 如果分箱失败（例如所有值相同），则统一返回 0
        return pd.Series([0] * len(series), index=series.index)

def remove_original_columns_if_binned(df):
    """
    对于每个 *_bin 列，如果该分箱列有超过一个 unique value，
    则认为分箱成功，自动删除与其同名（去除后缀 _bin）的原始列。
    """
    # 先收集所有以 `_bin` 结尾的列
    binned_cols = [c for c in df.columns if c.endswith('_bin')]
    
    for bin_col in binned_cols:
        # 如果分箱列实际分出了多个箱（unique values）
        if df[bin_col].nunique() > 1:
            # 构造原始特征列名
            orig_col = bin_col[:-4]  # 去掉 "_bin"
            # 如果原始列存在，则删除它
            if orig_col in df.columns:
                df.drop(columns=[orig_col], inplace=True)
                
    return df