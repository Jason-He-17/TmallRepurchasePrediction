# rfm_repurchase_feature.py

import numpy as np
import pandas as pd

def add_rfm_and_repurchase_features(matrix, origin_data):
    user_log = origin_data.user_log_format1.copy()
    # Ensure timestamp is of datetime type
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d')
    user_log = user_log[user_log['action_type'] == 'purchase'].copy()
    
    # ============ RFM Basic Features ============
    # Recency: Number of days since the last purchase
    max_date = user_log['time_stamp'].max()
    rfm = user_log.groupby(['user_id', 'merchant_id']).agg({
        'time_stamp': ['max', 'count']
    }).reset_index()
    rfm.columns = ['user_id', 'merchant_id', 'last_purchase_time', 'purchase_count']
    rfm['R'] = (max_date - rfm['last_purchase_time']).dt.days
    
    # Frequency: Here we use the number of transactions as an example
    rfm['F'] = rfm['purchase_count']
    
    # Monetary: Since there's no transaction amount available, we can use the total number of items purchased from the same store as an approximation.
    # Or estimate simply by counting the unique number of item_ids.
    rfm['M'] = rfm['purchase_count']  # Simplified
    
    # ============ Repurchase Count and Interval Features ============
    # Calculate statistics for multiple purchases by users at the store
    user_log = user_log.sort_values(['user_id', 'merchant_id', 'time_stamp'])
    user_log['repurchase_gap'] = user_log.groupby(['user_id','merchant_id'])['time_stamp'].diff().dt.days
    gap_stats = user_log.groupby(['user_id','merchant_id'])['repurchase_gap'].agg(['mean','std','min','max']).reset_index()
    gap_stats.rename(columns={
        'mean': 'repurchase_gap_mean',
        'std': 'repurchase_gap_std',
        'min': 'repurchase_gap_min',
        'max': 'repurchase_gap_max'
    }, inplace=True)
    
    # Merge RFM & gap_stats
    rfm = rfm.merge(gap_stats, how='left', on=['user_id','merchant_id'])
    
    # Merge into training matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(rfm.drop(['last_purchase_time','purchase_count'], axis=1),
                                                             on=['user_id','merchant_id'], how='left')
    # Fill missing values
    for col in ['R','F','M','repurchase_gap_mean','repurchase_gap_std','repurchase_gap_min','repurchase_gap_max']:
        matrix.train_test_matrix[col] = matrix.train_test_matrix[col].fillna(0)