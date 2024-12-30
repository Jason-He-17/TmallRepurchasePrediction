# rfm_repurchase_feature.py

import numpy as np
import pandas as pd

def add_rfm_and_repurchase_features(matrix, origin_data):
    user_log = origin_data.user_log_format1.copy()
    # 确保时间戳为datetime类型
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d')
    user_log = user_log[user_log['action_type'] == 'purchase'].copy()
    
    # ============ RFM 基础特征 ============
    # Recency：距离最后一次购买的天数
    max_date = user_log['time_stamp'].max()
    rfm = user_log.groupby(['user_id', 'merchant_id']).agg({
        'time_stamp': ['max', 'count']
    }).reset_index()
    rfm.columns = ['user_id', 'merchant_id', 'last_purchase_time', 'purchase_count']
    rfm['R'] = (max_date - rfm['last_purchase_time']).dt.days
    
    # Frequency：这里用交易次数示例
    rfm['F'] = rfm['purchase_count']
    
    # Monetary：此处没有金额可用，可用同一店铺下总购买商品件数作为近似
    # 或者统计 item_id 的去重数做简单估计
    rfm['M'] = rfm['purchase_count']  # 简化
    
    # ============ 复购次数和间隔特征 ============
    # 统计用户与店铺购买多次情况
    user_log = user_log.sort_values(['user_id', 'merchant_id', 'time_stamp'])
    user_log['repurchase_gap'] = user_log.groupby(['user_id','merchant_id'])['time_stamp'].diff().dt.days
    gap_stats = user_log.groupby(['user_id','merchant_id'])['repurchase_gap'].agg(['mean','std','min','max']).reset_index()
    gap_stats.rename(columns={
        'mean': 'repurchase_gap_mean',
        'std': 'repurchase_gap_std',
        'min': 'repurchase_gap_min',
        'max': 'repurchase_gap_max'
    }, inplace=True)
    
    # 合并 RFM & gap_stats
    rfm = rfm.merge(gap_stats, how='left', on=['user_id','merchant_id'])
    
    # 合并进训练矩阵
    matrix.train_test_matrix = matrix.train_test_matrix.merge(rfm.drop(['last_purchase_time','purchase_count'], axis=1),
                                                             on=['user_id','merchant_id'], how='left')
    # 缺失值填充
    for col in ['R','F','M','repurchase_gap_mean','repurchase_gap_std','repurchase_gap_min','repurchase_gap_max']:
        matrix.train_test_matrix[col] = matrix.train_test_matrix[col].fillna(0)
    
    # 删除M列
    matrix.train_test_matrix.drop(['M'], axis=1, inplace=True)