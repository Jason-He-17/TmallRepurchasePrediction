# um_feature.py

import pandas as pd
import numpy as np

from helper import _cap_values, _bin_values

def add_user_merchant_features(matrix, origin_data):
    """
    Add features related to user-merchant combinations to the training and testing matrix,
    and apply capping and binning to capture non-linear relationships.
    """
    user_log = origin_data.user_log_format1.copy()
    um_group = user_log.groupby(['user_id', 'merchant_id'])

    # ------------------------
    # Basic Statistical Features
    # ------------------------
    unique_counts = um_group.agg({
        'item_id': 'nunique',
        'cat_id': 'nunique',
        'brand_id': 'nunique'
    }).rename(columns={
        'item_id': 'um_iid',   # Number of unique item IDs for each user-merchant pair
        'cat_id': 'um_cid',    # Number of unique category IDs for each user-merchant pair
        'brand_id': 'um_bid'   # Number of unique brand IDs for each user-merchant pair
    })

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        unique_counts.reset_index(),
        on=['user_id', 'merchant_id'], how='left'
    )

    # Action counts
    action_counts = user_log.pivot_table(
        index=['user_id', 'merchant_id'],
        columns='action_type',
        aggfunc='size',
        fill_value=0
    ).reset_index().rename(columns={
        'click': 'um_click',            # Clicks per user-merchant pair
        'add-to-cart': 'um_cart',       # Add-to-cart actions per user-merchant pair
        'purchase': 'um_purchase',      # Purchases per user-merchant pair
        'add-to-favorite': 'um_fav'     # Favorites added per user-merchant pair
    })

    # Remove unnecessary columns (only remove 'um_cart', keep 'um_fav' for later use)
    action_counts.drop(['um_cart'], axis=1, inplace=True, errors='ignore')

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        action_counts,
        on=['user_id', 'merchant_id'], how='left'
    )

    # ------------------------
    # Activity Duration
    # ------------------------
    um_time = um_group['time_stamp'].agg(['min', 'max']).reset_index()
    um_time['um_days_between'] = (um_time['max'] - um_time['min']).dt.days
    um_time['um_days_between'] = um_time['um_days_between'].fillna(0)

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        um_time[['user_id', 'merchant_id', 'um_days_between']],
        on=['user_id', 'merchant_id'], how='left'
    )

    # Average orders per day
    matrix.train_test_matrix['um_avg_orders_per_day'] = (
        matrix.train_test_matrix['um_purchase'] / matrix.train_test_matrix['um_days_between'].replace(0, 1)
    )

    # ------------------------
    # Ratio Features Calculation
    # ------------------------
    matrix.train_test_matrix['um_purchase_per_click'] = (
        matrix.train_test_matrix['um_purchase'] / matrix.train_test_matrix['um_click']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    matrix.train_test_matrix['um_fav_per_click'] = (
        matrix.train_test_matrix['um_fav'] / matrix.train_test_matrix['um_click']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # ------------------------
    # Interaction Frequency Feature
    # ------------------------
    user_total_interactions = user_log.groupby('user_id').size().reset_index(name='user_total_interactions')
    um_features = um_group.size().reset_index(name='um_count')
    um_features = um_features.merge(user_total_interactions, on='user_id', how='left')
    um_features['um_interaction_freq'] = um_features['um_count'] / um_features['user_total_interactions']
    um_features['um_interaction_freq'] = um_features['um_interaction_freq'].replace([np.inf, -np.inf], 0).fillna(0)

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        um_features[['user_id', 'merchant_id', 'um_interaction_freq']],
        on=['user_id', 'merchant_id'], how='left'
    )

    # ------------------------
    # Capping and Binning Processing
    # ------------------------
    # List features that need capping and binning
    features_to_cap = [
        'um_iid', 'um_cid', 'um_bid',
        'um_click', 'um_purchase',
        'um_days_between', 'um_avg_orders_per_day',
        'um_purchase_per_click', 'um_fav_per_click',
        'um_interaction_freq'
    ]

    # Cap at 95th percentile first
    for col in features_to_cap:
        if col in matrix.train_test_matrix.columns:
            matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)

    # Then bin all numeric features that need binning
    features_to_bin = [
        'um_days_between',
        'um_avg_orders_per_day',
        'um_purchase_per_click',
        'um_fav_per_click',
        'um_interaction_freq'
    ]

    for col in features_to_bin:
        if col in matrix.train_test_matrix.columns:
            bin_col_name = f"{col}_bin"
            matrix.train_test_matrix[bin_col_name] = _bin_values(matrix.train_test_matrix[col], bins=8)

    # matrix.train_test_matrix.drop(columns=features_to_cap, inplace=True, errors='ignore')

def add_user_merchant_cart_purchase_interval(matrix, origin_data):
    """
    Calculate the average time interval between 'add-to-cart' and 'purchase' for each user-merchant pair,
    and add it as a feature to the training and testing matrix.
    """
    user_log = origin_data.user_log_format1.copy()

    # Retain only 'add-to-cart' and 'purchase' actions
    cart_purchase_log = user_log[user_log['action_type'].isin(['add-to-cart', 'purchase'])].copy()

    # Ensure 'time_stamp' is of datetime type
    if cart_purchase_log['time_stamp'].dtype != 'datetime64[ns]':
        cart_purchase_log['time_stamp'] = pd.to_datetime(
            '2016' + cart_purchase_log['time_stamp'].astype(str), format='%Y%m%d', errors='coerce'
        )

    # Sort to ensure chronological order
    cart_purchase_log = cart_purchase_log.sort_values(['user_id', 'merchant_id', 'time_stamp'])

    # Mark whether the action is a purchase
    cart_purchase_log['is_purchase'] = (cart_purchase_log['action_type'] == 'purchase').astype(int)

    # Use shift to find the next 'purchase' time after each 'add-to-cart'
    cart_purchase_log['next_purchase_time'] = cart_purchase_log.groupby(['user_id', 'merchant_id'])['time_stamp'].shift(-1)
    cart_purchase_log['next_action'] = cart_purchase_log.groupby(['user_id', 'merchant_id'])['action_type'].shift(-1)

    # Keep records where 'add-to-cart' is immediately followed by 'purchase'
    cart_purchase_log = cart_purchase_log[
        (cart_purchase_log['action_type'] == 'add-to-cart') &
        (cart_purchase_log['next_action'] == 'purchase')
    ]

    # Calculate the time interval (in days)
    cart_purchase_log['cart_purchase_interval_days'] = (
        cart_purchase_log['next_purchase_time'] - cart_purchase_log['time_stamp']
    ).dt.days

    # Calculate the average time interval for each user-merchant pair
    cart_purchase_interval = cart_purchase_log.groupby(['user_id', 'merchant_id'])['cart_purchase_interval_days'].mean().reset_index().rename(
        columns={'cart_purchase_interval_days': 'um_avg_cart_purchase_interval'}
    )

    # Merge the average time interval into the training and testing matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        cart_purchase_interval,
        on=['user_id', 'merchant_id'], how='left'
    )

    # Fill missing values (no corresponding 'add-to-cart' -> 'purchase' actions)
    matrix.train_test_matrix['um_avg_cart_purchase_interval'] \
        = matrix.train_test_matrix['um_avg_cart_purchase_interval'].fillna(-1)

    # Apply capping and binning to 'um_avg_cart_purchase_interval'
    if 'um_avg_cart_purchase_interval' in matrix.train_test_matrix.columns:
        # Capping
        matrix.train_test_matrix['um_avg_cart_purchase_interval'] = _cap_values(
            matrix.train_test_matrix['um_avg_cart_purchase_interval'], upper_percentile=95
        )
        # Binning
        matrix.train_test_matrix['um_avg_cart_purchase_interval_bin'] = _bin_values(
            matrix.train_test_matrix['um_avg_cart_purchase_interval'], bins=8
        )