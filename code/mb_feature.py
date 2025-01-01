# mb_feature.py

import numpy as np
import pandas as pd

from helper import _cap_values, _bin_values

def add_merchant_brand_features(matrix, origin_data):
    """
    Aggregate statistics for (merchant_id, brand_id) relationships, calculate interaction metrics for each merchant across brands, and merge features based on the most common brand_id.
    """
    user_log = origin_data.user_log_format1.copy()
    
    # Ensure time_stamp is of datetime type
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        # If originally in string format like '20161111', use format='%Y%m%d'
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']

    # 1) Group by merchant and brand to aggregate various action counts
    mb_group = user_log.groupby(['merchant_id', 'brand_id']).agg({
        'user_id': 'nunique',  
        'item_id': 'nunique',
        'action_type': [
            lambda x: (x == 'click').sum(),
            lambda x: (x == 'purchase').sum(),
            lambda x: (x == 'add-to-cart').sum(),
            lambda x: (x == 'add-to-favorite').sum()
        ]
    })

    # Rename columns
    mb_group.columns = [
        'mb_unique_users', 
        'mb_unique_items',
        'mb_click', 
        'mb_purchase', 
        'mb_cart', 
        'mb_fav'
    ]
    mb_group.reset_index(inplace=True)

    # 2) Calculate conversion rate features
    mb_group['mb_purchase_click_rate'] = (
        mb_group['mb_purchase'] / mb_group['mb_click'].replace(0, 1)
    )

    # 3) Capping & binning
    features_to_cap = ['mb_click','mb_purchase','mb_cart','mb_fav','mb_unique_users','mb_unique_items','mb_purchase_click_rate']
    for col in features_to_cap:
        if col in mb_group.columns:
            mb_group[col] = _cap_values(mb_group[col], upper_percentile=95)

    for col in ['mb_purchase_click_rate']:
        bin_col = f'{col}_bin'
        if col in mb_group.columns:
            mb_group[bin_col] = _bin_values(mb_group[col], bins=8)

    # 4) Find the most common brand_id for each merchant_id
    mb_top = user_log.groupby(['merchant_id', 'brand_id']).size().reset_index(name='freq_brand')
    mb_top = mb_top.sort_values(['merchant_id', 'freq_brand'], ascending=[True, False])
    mb_top = mb_top.drop_duplicates(subset=['merchant_id'], keep='first')
    mb_top.rename(columns={'brand_id': 'top_brand_id'}, inplace=True)

    # 5) Merge mb_group with mb_top to get features for the most common brand_id per merchant_id
    mb_features = mb_top.merge(mb_group, left_on=['merchant_id', 'top_brand_id'], right_on=['merchant_id', 'brand_id'], how='left')
    mb_features.drop(['brand_id', 'freq_brand'], axis=1, inplace=True)

    # 6) Merge into matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        mb_features.drop('top_brand_id', axis=1),
        on='merchant_id',
        how='left'
    )

    # 7) Fill missing values for numerical columns
    numerical_cols = mb_features.columns.drop(['merchant_id', 'top_brand_id'])
    numerical_cols = [col for col in numerical_cols if col in matrix.train_test_matrix.columns]
    matrix.train_test_matrix[numerical_cols] = matrix.train_test_matrix[numerical_cols].fillna(0)