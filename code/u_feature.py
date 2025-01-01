# u_feature.py

import pandas as pd
import numpy as np

from helper import _cap_values, _bin_values

def add_user_features(matrix, origin_data):
    """
    Extend existing user features by adding binning and capping to improve the modeling of non-linear relationships.
    """
    user_log = origin_data.user_log_format1.copy()
    
    # Ensure 'time_stamp' is of datetime type
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')
    user_log['date'] = user_log['time_stamp'].dt.date
    
    user_group = user_log.groupby('user_id')
    
    # ------------------------
    # Basic Statistical Features
    # ------------------------
    unique_counts = user_group.agg({
        'item_id': 'nunique',
        'cat_id': 'nunique',
        'merchant_id': 'nunique',
        'brand_id': 'nunique'
    }).rename(columns={
        'item_id': 'u_iid',
        'cat_id': 'u_cid',
        'merchant_id': 'u_mid',
        'brand_id': 'u_bid'
    })
    
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        unique_counts.reset_index(),
        on='user_id', how='left'
    )
    
    # Action counts
    action_counts = user_log.pivot_table(
        index='user_id',
        columns='action_type',
        aggfunc='size',
        fill_value=0
    ).reset_index().rename(columns={
        'click': 'u_click',
        'add-to-cart': 'u_cart',
        'purchase': 'u_purchase',
        'add-to-favorite': 'u_fav'
    })
    
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        action_counts,
        on='user_id', how='left'
    )
    
    # ------------------------
    # Continuous Purchase Intervals
    # ------------------------
    purchase_logs = user_log[user_log['action_type'] == 'purchase'].copy()
    purchase_logs = purchase_logs.sort_values(['user_id', 'time_stamp'])
    purchase_logs['diff_days'] = purchase_logs.groupby('user_id')['time_stamp'].diff().dt.days
    
    avg_days_between_purchases = purchase_logs.groupby('user_id')['diff_days'].mean().reset_index().rename(
        columns={'diff_days': 'u_avg_days_between_purchases'}
    )
    avg_days_between_purchases['u_avg_days_between_purchases'] = avg_days_between_purchases['u_avg_days_between_purchases'].fillna(185)

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        avg_days_between_purchases,
        on='user_id', how='left'
    )
    
    # ------------------------
    # Ratios of Clicks/Cart/Add-to-Favorite/Purchase
    # ------------------------
    matrix.train_test_matrix['u_purchase_click_rate'] = (
        matrix.train_test_matrix['u_purchase'] / matrix.train_test_matrix['u_click']
    )
    matrix.train_test_matrix['u_cart_click_rate'] = (
        matrix.train_test_matrix['u_cart'] / matrix.train_test_matrix['u_click']
    )
    matrix.train_test_matrix['u_fav_click_rate'] = (
        matrix.train_test_matrix['u_fav'] / matrix.train_test_matrix['u_click']
    )
    
    # Replace infinity and NaN
    matrix.train_test_matrix[['u_purchase_click_rate','u_cart_click_rate','u_fav_click_rate']] = (
        matrix.train_test_matrix[['u_purchase_click_rate','u_cart_click_rate','u_fav_click_rate']]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    
    # Cart to Purchase Ratio
    matrix.train_test_matrix['u_purchase_cart_rate'] = (
        matrix.train_test_matrix['u_purchase'] / matrix.train_test_matrix['u_cart']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # ------------------------
    # Active Days & Purchase Days
    # ------------------------
    user_active_days = user_group['date'].nunique().reset_index().rename(
        columns={'date': 'u_active_days'}
    )
    matrix.train_test_matrix = matrix.train_test_matrix.merge(user_active_days, on='user_id', how='left')
    matrix.train_test_matrix['u_active_days'] = matrix.train_test_matrix['u_active_days'].fillna(0).astype(int)
    
    user_purchase_days = purchase_logs.groupby('user_id')['time_stamp'].nunique().reset_index().rename(
        columns={'time_stamp': 'u_purchase_days'}
    )
    matrix.train_test_matrix = matrix.train_test_matrix.merge(user_purchase_days, on='user_id', how='left')
    matrix.train_test_matrix['u_purchase_days'] = matrix.train_test_matrix['u_purchase_days'].fillna(0).astype(int)
    
    # List features that need capping and binning
    features_to_cap = [
        'u_click','u_purchase','u_cart','u_fav','u_iid','u_cid','u_bid','u_mid',
        'u_active_days','u_purchase_days','u_avg_days_between_purchases',
        'u_purchase_click_rate','u_cart_click_rate','u_fav_click_rate','u_purchase_cart_rate'
    ]
    
    # Cap at 95th percentile first
    for col in features_to_cap:
        if col in matrix.train_test_matrix.columns:
            matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)
    
    # Then bin all numeric features that need binning
    # Choose features to bin (based on business understanding and data distribution)
    features_to_bin = [
        'u_active_days',
        'u_purchase_days',
        'u_avg_days_between_purchases',
        'u_purchase_click_rate',
        'u_cart_click_rate',
        'u_fav_click_rate',
        'u_purchase_cart_rate'
    ]
    
    for col in features_to_bin:
        if col in matrix.train_test_matrix.columns:
            bin_col_name = f"{col}_bin"
            matrix.train_test_matrix[bin_col_name] = _bin_values(matrix.train_test_matrix[col], bins=8)

    # matrix.train_test_matrix.drop(columns=features_to_cap, inplace=True, errors='ignore')

def add_user_1111_features(matrix, origin_data):
    """
    Add interaction features for user activities on 11/11, including purchase frequency and its proportion, with capping and binning.
    """
    user_log = origin_data.user_log_format1.copy()

    # Ensure time_stamp is of datetime type
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    # Mark 11/11
    user_log['is_1111'] = (user_log['time_stamp'].dt.month == 11) & (user_log['time_stamp'].dt.day == 11)

    # 1. Count each user's various actions on 11/11
    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']
    action_1111 = user_log[user_log['is_1111']].groupby(['user_id', 'action_type']).size().unstack(fill_value=0).reset_index()
    action_1111.columns = ['user_id'] + [f'u_{action}_1111' for action in actions]

    # 2. Count each user's total various actions
    action_total = user_log.groupby(['user_id', 'action_type']).size().unstack(fill_value=0).reset_index()
    action_total.columns = ['user_id'] + [f'u_{action}_total' for action in actions]

    # 3. Merge 11/11 action counts with total action counts
    user_1111 = action_1111.merge(action_total, on='user_id', how='left')

    # 4. Fill missing values with 0
    user_1111.fillna(0, inplace=True)

    # 5. Calculate 11/11 action proportions
    for action in actions:
        action_1111_col = f'u_{action}_1111'
        action_total_col = f'u_{action}_total'
        ratio_col = f'u_{action}_1111_ratio'
        user_1111[ratio_col] = user_1111[action_1111_col] / user_1111[action_total_col].replace(0, 1)
    
    # 6. Perform capping
    features_to_cap = [f'u_{action}_1111' for action in actions] + [f'u_{action}_1111_ratio' for action in actions]
    for col in features_to_cap:
        if col in user_1111.columns:
            user_1111[col] = _cap_values(user_1111[col], upper_percentile=95)

    # 7. Perform binning
    features_to_bin = [f'u_{action}_1111_ratio' for action in actions]
    for col in features_to_bin:
        if col in user_1111.columns:
            bin_col = f"{col}_bin"
            user_1111[bin_col] = _bin_values(user_1111[col], bins=8)

    # 8. Select features to merge into train_test_matrix
    merge_cols = [f'u_{action}_1111' for action in actions] + \
                 [f'u_{action}_1111_ratio' for action in actions] + \
                 [f'u_{action}_1111_ratio_bin' for action in actions]

    # 9. Merge features into train_test_matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        user_1111[['user_id'] + merge_cols],
        on='user_id',
        how='left'
    )

    # 10. Handle possible missing values after merging
    matrix.train_test_matrix[merge_cols] = matrix.train_test_matrix[merge_cols].fillna(0)