# m_feature.py

import pandas as pd
import numpy as np

from helper import _cap_values, _bin_values

def add_merchant_features(matrix, origin_data):
    """
    添加与商家相关的特征到训练和测试矩阵中，并进行截断与分箱处理以捕捉非线性关系。
    """
    user_log = origin_data.user_log_format1.copy()
    
    # 确保 'time_stamp' 是 datetime 类型
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')
    
    merchant_group = user_log.groupby('merchant_id')
    
    # ------------------------
    # 基础统计特征
    # ------------------------
    unique_counts = merchant_group.agg({
        'item_id': 'nunique',
        'cat_id': 'nunique',
        'user_id': 'nunique',
        'brand_id': 'nunique'
    }).rename(columns={
        'item_id': 'm_iid',    # 每个商家的唯一商品ID数量
        'cat_id': 'm_cid',     # 每个商家的唯一品类ID数量
        'user_id': 'm_uid',    # 每个商家的唯一用户ID数量
        'brand_id': 'm_bid'    # 每个商家的唯一品牌ID数量
    })
    
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        unique_counts.reset_index(),
        on='merchant_id', how='left'
    )
    
    # 行为计数
    action_counts = user_log.pivot_table(
        index='merchant_id',
        columns='action_type',
        aggfunc='size',
        fill_value=0
    ).reset_index().rename(columns={
        'click': 'm_click',          # 每个商家的点击次数
        'add-to-cart': 'm_cart',     # 每个商家的加入购物车次数
        'purchase': 'm_purchase',    # 每个商家的购买次数
        'add-to-favorite': 'm_fav'   # 每个商家的收藏次数
    })
    
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        action_counts,
        on='merchant_id', how='left'
    )
    
    # ------------------------
    # 活动持续时间
    # ------------------------
    merchant_time = merchant_group['time_stamp'].agg(['min', 'max']).reset_index()
    merchant_time['m_days_between'] = (merchant_time['max'] - merchant_time['min']).dt.days
    merchant_time['m_days_between'] = merchant_time['m_days_between'].fillna(0)
    
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        merchant_time[['merchant_id', 'm_days_between']],
        on='merchant_id', how='left'
    )
    
    # 平均每日订单数
    matrix.train_test_matrix['m_avg_orders_per_day'] = (
        matrix.train_test_matrix['m_purchase'] / matrix.train_test_matrix['m_days_between'].replace(0, 1)
    )
    
    # ------------------------
    # 计算比率特征
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
    # 截断与分箱处理
    # ------------------------
    # 列出需要截断和分箱的特征
    features_to_cap = [
        'm_click', 'm_purchase', 'm_cart', 'm_fav',
        'm_iid', 'm_cid', 'm_bid', 'm_uid',
        'm_days_between', 'm_avg_orders_per_day',
        'm_purchase_per_click', 'm_cart_per_click', 'm_fav_per_click'
    ]
    
    # 先按 99% 分位截断
    for col in features_to_cap:
        if col in matrix.train_test_matrix.columns:
            matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)
    
    # 然后对所有需要分箱的数值特征进行分箱
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
    针对 merchant_id 计算 11/11 行为特征并合并。
    """
    user_log = origin_data.user_log_format1.copy()
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    user_log['is_1111'] = (user_log['time_stamp'].dt.month == 11) & (user_log['time_stamp'].dt.day == 11)
    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']

    # 1) 统计 11/11 行为
    m_1111 = user_log[user_log['is_1111']].groupby(['merchant_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:  # 确保列完整
        if act not in m_1111.columns:
            m_1111[act] = 0
    m_1111.columns = [f'm_{a}_1111' for a in actions]
    m_1111.reset_index(inplace=True)

    # 2) 统计总行为
    m_total = user_log.groupby(['merchant_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in m_total.columns:
            m_total[act] = 0
    m_total.columns = [f'm_{a}_total' for a in actions]
    m_total.reset_index(inplace=True)

    # 3) 合并并计算占比
    features = m_1111.merge(m_total, on='merchant_id', how='left').fillna(0)
    for act in actions:
        features[f'm_{act}_1111_ratio'] = (
            features[f'm_{act}_1111'] / features[f'm_{act}_total'].replace(0, 1)
        )

    # 4) 截断与分箱
    for col in features.columns:
        if col.startswith('m_') and col != 'merchant_id':
            features[col] = _cap_values(features[col], upper_percentile=95)

    for act in actions:
        ratio_col = f'm_{act}_1111_ratio'
        if ratio_col in features.columns:
            bin_col = f'{ratio_col}_bin'
            features[bin_col] = _bin_values(features[ratio_col], bins=8)

    # 5) 合并到 matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(features, on='merchant_id', how='left')

    # 6) 填充缺失值，仅针对数值型列
    numerical_cols = features.columns.drop('merchant_id')
    numerical_cols = list(numerical_cols)
    matrix.train_test_matrix[numerical_cols] = matrix.train_test_matrix[numerical_cols].fillna(0)