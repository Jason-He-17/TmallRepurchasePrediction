# m_feature.py

import pandas as pd
import numpy as np

from helper import _cap_values, _bin_values

def add_merchant_features(matrix, origin_data):
    """
    Add merchant-related features to the training and testing matrix and perform capping and binning to capture non-linear relationships.
    """
    user_log = origin_data.user_log_format1.copy()
    
    # Ensure 'time_stamp' is of datetime type
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')
    
    merchant_group = user_log.groupby('merchant_id')
    
    # ------------------------
    # Basic statistical features
    # ------------------------
    unique_counts = merchant_group.agg({
        'item_id': 'nunique',
        'cat_id': 'nunique',
        'user_id': 'nunique',
        'brand_id': 'nunique'
    }).rename(columns={
        'item_id': 'm_iid',    # Number of unique item IDs per merchant
        'cat_id': 'm_cid',     # Number of unique category IDs per merchant
        'user_id': 'm_uid',    # Number of unique user IDs per merchant
        'brand_id': 'm_bid'    # Number of unique brand IDs per merchant
    })
    
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        unique_counts.reset_index(),
        on='merchant_id', how='left'
    )
    
    # Action counts
    action_counts = user_log.pivot_table(
        index='merchant_id',
        columns='action_type',
        aggfunc='size',
        fill_value=0
    ).reset_index().rename(columns={
        'click': 'm_click',          # Number of clicks per merchant
        'add-to-cart': 'm_cart',     # Number of add-to-cart actions per merchant
        'purchase': 'm_purchase',    # Number of purchases per merchant
        'add-to-favorite': 'm_fav'   # Number of add-to-favorite actions per merchant
    })
    
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        action_counts,
        on='merchant_id', how='left'
    )
    
    # ------------------------
    # Duration of activity
    # ------------------------
    merchant_time = merchant_group['time_stamp'].agg(['min', 'max']).reset_index()
    merchant_time['m_days_between'] = (merchant_time['max'] - merchant_time['min']).dt.days
    merchant_time['m_days_between'] = merchant_time['m_days_between'].fillna(0)
    
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        merchant_time[['merchant_id', 'm_days_between']],
        on='merchant_id', how='left'
    )
    
    # Average number of orders per day
    matrix.train_test_matrix['m_avg_orders_per_day'] = (
        matrix.train_test_matrix['m_purchase'] / matrix.train_test_matrix['m_days_between'].replace(0, 1)
    )
    
    # ------------------------
    # Calculate ratio features
    # ------------------------
    matrix.train_test_matrix['m_purchase_per_click'] = (
        matrix.train_test_matrix['m_purchase'] / matrix.train_test_matrix['m_click']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    matrix.train_test_matrix['m_cart_per_click'] = (
        matrix.train_test_matrix['m_cart'] / matrix.train_test_matrix['m_click']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    matrix.train_test_matrix['m_fav_per_click'] = (
        matrix.train_test_matrix['m_fav'] / matrix.train_test_matrix['m_click']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # ------------------------
    # Capping and binning
    # ------------------------
    # List features that need capping and binning
    features_to_cap = [
        'm_click', 'm_purchase', 'm_cart', 'm_fav',
        'm_iid', 'm_cid', 'm_bid', 'm_uid',
        'm_days_between', 'm_avg_orders_per_day',
        'm_purchase_per_click', 'm_cart_per_click', 'm_fav_per_click'
    ]
    
    # Cap at the 95th percentile first
    for col in features_to_cap:
        if col in matrix.train_test_matrix.columns:
            matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)
    
    # Then bin all numerical features that need binning
    features_to_bin = [
        'm_days_between',
        'm_avg_orders_per_day',
        'm_purchase_per_click',
        'm_cart_per_click',
        'm_fav_per_click'
    ]
    
    for col in features_to_bin:
        if col in matrix.train_test_matrix.columns:
            bin_col_name = f"{col}_bin"
            matrix.train_test_matrix[bin_col_name] = _bin_values(matrix.train_test_matrix[col], bins=8)

    # matrix.train_test_matrix.drop(columns=features_to_cap, inplace=True, errors='ignore')

def add_merchant_1111_features(matrix, origin_data):
    """
    Calculate behavior features for 11/11 (Singles' Day) for each merchant_id and merge them.
    """
    user_log = origin_data.user_log_format1.copy()
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    user_log['is_1111'] = (user_log['time_stamp'].dt.month == 11) & (user_log['time_stamp'].dt.day == 11)
    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']

    # 1) Statistic of 11/11 behavior
    m_1111 = user_log[user_log['is_1111']].groupby(['merchant_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:  # Ensure completeness of columns
        if act not in m_1111.columns:
            m_1111[act] = 0
    m_1111.columns = [f'm_{a}_1111' for a in actions]
    m_1111.reset_index(inplace=True)

    # 2) Statistic of total behavior
    m_total = user_log.groupby(['merchant_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in m_total.columns:
            m_total[act] = 0
    m_total.columns = [f'm_{a}_total' for a in actions]
    m_total.reset_index(inplace=True)

    # 3) Merge and calculate ratios
    features = m_1111.merge(m_total, on='merchant_id', how='left').fillna(0)
    for act in actions:
        features[f'm_{act}_1111_ratio'] = (
            features[f'm_{act}_1111'] / features[f'm_{act}_total'].replace(0, 1)
        )

    # 4) Capping and binning
    for col in features.columns:
        if col.startswith('m_') and col != 'merchant_id':
            features[col] = _cap_values(features[col], upper_percentile=95)

    for act in actions:
        ratio_col = f'm_{act}_1111_ratio'
        if ratio_col in features.columns:
            bin_col = f'{ratio_col}_bin'
            features[bin_col] = _bin_values(features[ratio_col], bins=8)

    # 5) Merge into matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(features, on='merchant_id', how='left')

    # 6) Fill missing values, only for numeric columns
    numerical_cols = features.columns.drop('merchant_id')
    numerical_cols = list(numerical_cols)
    matrix.train_test_matrix[numerical_cols] = matrix.train_test_matrix[numerical_cols].fillna(0)