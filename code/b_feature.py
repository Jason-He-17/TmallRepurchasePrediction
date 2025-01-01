# b_feature.py

import numpy as np
import pandas as pd

from helper import _cap_values, _bin_values

def add_brand_features(matrix, origin_data):
    """
    1. Aggregate statistics on the global brand_id to obtain statistical information on brand features.
    2. Group by (user_id, merchant_id, brand_id) and select the most common brand_id as top_brand_id.
    3. Merge the statistical information of the most common brand_id to the (user_id, merchant_id) level and then merge it into the training matrix.
    4. Perform a uniform 95% percentile capping + 8 binning and remove unused datetime columns.
    """
    user_log = origin_data.user_log_format1.copy()

    # Ensure time_stamp is of datetime type
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        # If originally in string format like '20161111', use format='%Y%m%d'
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    #################################
    # 1) Global brand feature aggregation: brand_id  #
    #################################
    brand_group = user_log.groupby('brand_id').agg({
        'merchant_id': 'nunique',
        'item_id': 'nunique',
        'time_stamp': ['min', 'max'],
        # Count purchase actions
        'action_type': lambda x: (x == 'purchase').sum()
    })
    brand_group.columns = [
        'b_unique_merchants', 'b_unique_items',
        'b_min_time', 'b_max_time', 'b_purchase_count'
    ]
    brand_group.reset_index(inplace=True)

    # Calculate duration in days
    brand_group['b_days_between'] = (
        brand_group['b_max_time'] - brand_group['b_min_time']
    ).dt.days.fillna(0)

    # Calculate average purchase frequency (per day)
    brand_group['b_avg_purchases_per_day'] = brand_group.apply(
        lambda row: row['b_purchase_count'] / row['b_days_between'] if row['b_days_between'] > 0 else 0,
        axis=1
    )

    # Remove datetime columns that are no longer needed
    brand_group.drop(['b_min_time', 'b_max_time'], axis=1, inplace=True)

    # Statistics for each action type by brand_id in user_log
    brand_action_counts = user_log.pivot_table(
        index='brand_id', columns='action_type', aggfunc='size', fill_value=0
    ).reset_index().rename(columns={
        'click': 'b_click',
        'add-to-cart': 'b_cart',
        'purchase': 'b_purchase',
        'add-to-favorite': 'b_fav'
    })
    # Merge with brand_group
    brand_group = brand_group.merge(brand_action_counts, on='brand_id', how='left')

    # Calculate ratios
    brand_group['b_purchase_click_rate'] = brand_group.apply(
        lambda row: row['b_purchase'] / row['b_click'] if row['b_click'] > 0 else 0, axis=1
    )
    brand_group['b_cart_click_rate'] = brand_group.apply(
        lambda row: row['b_cart'] / row['b_click'] if row['b_click'] > 0 else 0, axis=1
    )
    brand_group['b_fav_click_rate'] = brand_group.apply(
        lambda row: row['b_fav'] / row['b_click'] if row['b_click'] > 0 else 0, axis=1
    )

    ###################################
    # 2) Find the most common brand_id in user_id, merchant_id => top_brand_id
    ###################################
    umb = user_log.groupby(['user_id', 'merchant_id', 'brand_id']).size().reset_index(name='freq_brand')
    # Sort by freq_brand descending, then deduplicate by (user_id, merchant_id) => get the most common brand_id
    umb.sort_values('freq_brand', ascending=False, inplace=True)
    umb.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first', inplace=True)
    umb.rename(columns={'brand_id': 'top_brand_id'}, inplace=True)

    # Concatenate with brand_group
    umb = umb.merge(brand_group, left_on='top_brand_id', right_on='brand_id', how='left').drop('brand_id', axis=1)

    #####################################################
    # 3) Concatenate the statistical information of the most common top_brand_id back into the training matrix
    #####################################################
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        umb, on=['user_id', 'merchant_id'], how='left'
    )

    ###################################
    # 4) Uniformly perform 95% percentile capping + 8 binning
    ###################################
    # Select columns to cap (all numeric columns starting with b_ need capping)
    features_to_cap = [
        'b_unique_merchants', 'b_unique_items', 'b_purchase_count',
        'b_days_between', 'b_avg_purchases_per_day',
        'b_purchase', 'b_click', 'b_cart', 'b_fav',
        'b_purchase_click_rate', 'b_cart_click_rate', 'b_fav_click_rate',
        'freq_brand'
    ]
    for col in features_to_cap:
        if col in matrix.train_test_matrix.columns:
            matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)

    # Select columns to bin (based on settings from other files, use 8 bins)
    features_to_bin = [
        'b_days_between', 'b_avg_purchases_per_day',
        'b_purchase_click_rate', 'b_cart_click_rate', 'b_fav_click_rate'
    ]
    for col in features_to_bin:
        if col in matrix.train_test_matrix.columns:
            bin_col = f'{col}_bin'
            matrix.train_test_matrix[bin_col] = _bin_values(matrix.train_test_matrix[col], bins=8)

    # Finally completed, no need to remove datetime columns because they were removed in step 1

def add_brand_1111_features(matrix, origin_data):
    """
    Create 11/11 specific features for brands, first find the most common brand_id for (user_id, merchant_id), then merge the brand_1111 features.
    """
    user_log = origin_data.user_log_format1.copy()
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    user_log['is_1111'] = (user_log['time_stamp'].dt.month == 11) & (user_log['time_stamp'].dt.day == 11)
    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']

    # 1) Statistic actions for brand on 11/11
    b_1111 = user_log[user_log['is_1111']].groupby(['brand_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in b_1111.columns:
            b_1111[act] = 0
    b_1111.columns = [f'b_{a}_1111' for a in actions]
    b_1111.reset_index(inplace=True)

    # 2) Statistic total actions for brand
    b_total = user_log.groupby(['brand_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in b_total.columns:
            b_total[act] = 0
    b_total.columns = [f'b_{a}_total' for a in actions]
    b_total.reset_index(inplace=True)

    # 3) Merge and calculate proportions
    brand_1111 = b_1111.merge(b_total, on='brand_id', how='left').fillna(0)
    for act in actions:
        brand_1111[f'b_{act}_1111_ratio'] = (
            brand_1111[f'b_{act}_1111'] / brand_1111[f'b_{act}_total'].replace(0, 1)
        )

    # 4) Capping & Binning
    for col in brand_1111.columns:
        if col.startswith('b_') and col != 'brand_id':
            brand_1111[col] = _cap_values(brand_1111[col], upper_percentile=95)

    for act in actions:
        ratio_col = f'b_{act}_1111_ratio'
        if ratio_col in brand_1111.columns:
            bin_col = f'{ratio_col}_bin'
            brand_1111[bin_col] = _bin_values(brand_1111[ratio_col], bins=8)

    # 5) Find the most common brand_id => top_brand_id (similar to add_brand_features)
    umb = user_log.groupby(['user_id', 'merchant_id', 'brand_id']).size().reset_index(name='freq_brand')
    umb.sort_values('freq_brand', ascending=False, inplace=True)
    umb.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first', inplace=True)
    umb.rename(columns={'brand_id': 'top_brand_id'}, inplace=True)

    # 6) Concatenate 1111 features
    umb = umb.merge(brand_1111, left_on='top_brand_id', right_on='brand_id', how='left').drop('brand_id', axis=1)

    # 7) Merge into matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        umb, on=['user_id', 'merchant_id'], how='left'
    )

    ###################################
    # 8) Fill only missing values in numerical columns
    ###################################
    numerical_cols = brand_1111.columns.drop(['brand_id'])
    numerical_cols = [col for col in numerical_cols if col in matrix.train_test_matrix.columns]
    matrix.train_test_matrix[numerical_cols] = matrix.train_test_matrix[numerical_cols].fillna(0)