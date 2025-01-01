# c_feature.py

import numpy as np
import pandas as pd

from helper import _cap_values, _bin_values

def add_category_features(matrix, origin_data):
    """
    1. Aggregate statistics on the global cat_id to obtain statistical information on category features.
    2. Group by (user_id, merchant_id, cat_id) and select the most common cat_id as top_cat_id.
    3. Merge the statistical information of the most common cat_id to the (user_id, merchant_id) level and then merge it into the training matrix.
    4. Perform a uniform 95% percentile capping + 8 binning and remove unused datetime columns.
    """
    user_log = origin_data.user_log_format1.copy()

    # Ensure time_stamp is of datetime type
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        # If originally in string format like '20161111', use format='%Y%m%d'
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    ##############################
    # 1) Global category feature aggregation: cat_id #
    ##############################
    cat_group = user_log.groupby('cat_id').agg({
        'user_id': 'nunique',
        'merchant_id': 'nunique',
        'item_id': 'nunique',
        'time_stamp': ['min', 'max'],
        # Count purchase actions
        'action_type': lambda x: (x == 'purchase').sum()
    })
    cat_group.columns = [
        'c_unique_users', 'c_unique_merchants', 'c_unique_items',
        'c_min_time', 'c_max_time', 'c_purchase_count'
    ]
    cat_group.reset_index(inplace=True)

    # Calculate duration in days
    cat_group['c_days_between'] = (
        cat_group['c_max_time'] - cat_group['c_min_time']
    ).dt.days.fillna(0)

    # Calculate average purchase frequency (per day)
    cat_group['c_avg_purchases_per_day'] = cat_group.apply(
        lambda row: row['c_purchase_count'] / row['c_days_between'] if row['c_days_between'] > 0 else 0,
        axis=1
    )

    # Remove datetime columns that are no longer needed
    cat_group.drop(['c_min_time', 'c_max_time'], axis=1, inplace=True)

    # Statistics for each action type by cat_id in user_log
    cat_action_counts = user_log.pivot_table(
        index='cat_id', columns='action_type', aggfunc='size', fill_value=0
    ).reset_index().rename(columns={
        'click': 'c_click',
        'add-to-cart': 'c_cart',
        'purchase': 'c_purchase',
        'add-to-favorite': 'c_fav'
    })
    # Merge with cat_group
    cat_group = cat_group.merge(cat_action_counts, on='cat_id', how='left')

    # Calculate ratios
    cat_group['c_purchase_click_rate'] = cat_group.apply(
        lambda row: row['c_purchase'] / row['c_click'] if row['c_click'] > 0 else 0, axis=1
    )
    cat_group['c_cart_click_rate'] = cat_group.apply(
        lambda row: row['c_cart'] / row['c_click'] if row['c_click'] > 0 else 0, axis=1
    )
    cat_group['c_fav_click_rate'] = cat_group.apply(
        lambda row: row['c_fav'] / row['c_click'] if row['c_click'] > 0 else 0, axis=1
    )

    ###############################
    # 2) Find the most common cat_id in user_id, merchant_id => top_cat_id
    ###############################
    umc = user_log.groupby(['user_id', 'merchant_id', 'cat_id']).size().reset_index(name='freq_cat')
    # Sort by freq_cat descending, then deduplicate by (user_id, merchant_id) => get the most common cat_id
    umc.sort_values('freq_cat', ascending=False, inplace=True)
    umc.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first', inplace=True)
    umc.rename(columns={'cat_id': 'top_cat_id'}, inplace=True)

    # Concatenate with cat_group
    umc = umc.merge(cat_group, left_on='top_cat_id', right_on='cat_id', how='left').drop('cat_id', axis=1)

    #####################################################
    # 3) Concatenate the statistical information of the most common top_cat_id back into the training matrix
    #####################################################
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        umc, on=['user_id', 'merchant_id'], how='left'
    )

    ###################################
    # 4) Uniformly perform 95% percentile capping + 8 binning
    ###################################
    # Select columns to cap (all numeric columns starting with c_ need capping)
    features_to_cap = [
        'c_unique_users', 'c_unique_merchants', 'c_unique_items', 'c_purchase_count',
        'c_days_between', 'c_avg_purchases_per_day',
        'c_purchase', 'c_click', 'c_cart', 'c_fav',
        'c_purchase_click_rate', 'c_cart_click_rate', 'c_fav_click_rate',
        'freq_cat'
    ]
    for col in features_to_cap:
        if col in matrix.train_test_matrix.columns:
            matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)

    # Select columns to bin (based on settings from other files, use 8 bins)
    features_to_bin = [
        'c_days_between', 'c_avg_purchases_per_day',
        'c_purchase_click_rate', 'c_cart_click_rate', 'c_fav_click_rate'
    ]
    for col in features_to_bin:
        if col in matrix.train_test_matrix.columns:
            bin_col = f'{col}_bin'
            matrix.train_test_matrix[bin_col] = _bin_values(matrix.train_test_matrix[col], bins=8)

    # Finally completed, no need to remove datetime columns because they were removed in step 1

def add_category_1111_features(matrix, origin_data):
    """
    Create 11/11 specific features for categories, first find the most common cat_id for (user_id, merchant_id), then merge the cat_1111 features.
    """
    user_log = origin_data.user_log_format1.copy()
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    user_log['is_1111'] = (user_log['time_stamp'].dt.month == 11) & (user_log['time_stamp'].dt.day == 11)
    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']

    # 1) Statistic actions for cat on 11/11
    c_1111 = user_log[user_log['is_1111']].groupby(['cat_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in c_1111.columns:
            c_1111[act] = 0
    c_1111.columns = [f'c_{a}_1111' for a in actions]
    c_1111.reset_index(inplace=True)

    # 2) Statistic total actions for cat
    c_total = user_log.groupby(['cat_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in c_total.columns:
            c_total[act] = 0
    c_total.columns = [f'c_{a}_total' for a in actions]
    c_total.reset_index(inplace=True)

    # 3) Merge and calculate proportions
    cat_1111 = c_1111.merge(c_total, on='cat_id', how='left').fillna(0)
    for act in actions:
        cat_1111[f'c_{act}_1111_ratio'] = (
            cat_1111[f'c_{act}_1111'] / cat_1111[f'c_{act}_total'].replace(0, 1)
        )

    # 4) Capping & Binning
    for col in cat_1111.columns:
        if col.startswith('c_') and col != 'cat_id':
            cat_1111[col] = _cap_values(cat_1111[col], upper_percentile=95)

    for act in actions:
        ratio_col = f'c_{act}_1111_ratio'
        if ratio_col in cat_1111.columns:
            bin_col = f'{ratio_col}_bin'
            cat_1111[bin_col] = _bin_values(cat_1111[ratio_col], bins=8)

    # 5) Find the most common cat_id => top_cat_id
    umc = user_log.groupby(['user_id', 'merchant_id', 'cat_id']).size().reset_index(name='freq_cat')
    umc.sort_values('freq_cat', ascending=False, inplace=True)
    umc.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first', inplace=True)
    umc.rename(columns={'cat_id': 'top_cat_id'}, inplace=True)

    # 6) Concatenate 1111 features
    umc = umc.merge(cat_1111, left_on='top_cat_id', right_on='cat_id', how='left').drop('cat_id', axis=1)

    # 7) Merge into matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        umc, on=['user_id', 'merchant_id'], how='left'
    )

    ###################################
    # 8) Fill only missing values in numerical columns
    ###################################
    numerical_cols = cat_1111.columns.drop(['cat_id'])
    numerical_cols = [col for col in numerical_cols if col in matrix.train_test_matrix.columns]
    matrix.train_test_matrix[numerical_cols] = matrix.train_test_matrix[numerical_cols].fillna(0)