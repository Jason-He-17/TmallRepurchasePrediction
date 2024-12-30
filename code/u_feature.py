# u_feature.py

import pandas as pd
import numpy as np

from helper import _cap_values, _bin_values

def add_user_features(matrix, origin_data):
    """
    对现有用户特征进行扩展，并加入分箱与截断，提高对非线性关系的刻画能力。
    """
    user_log = origin_data.user_log_format1.copy()
    
    # 确保 'time_stamp' 是 datetime 类型
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')
    user_log['date'] = user_log['time_stamp'].dt.date
    
    user_group = user_log.groupby('user_id')
    
    # ------------------------
    # 基础统计特征
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
    
    # 行为计数
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
    # 连续购买间隔
    # ------------------------
    purchase_logs = user_log[user_log['action_type'] == 'purchase'].copy()
    purchase_logs = purchase_logs.sort_values(['user_id', 'time_stamp'])
    purchase_logs['diff_days'] = purchase_logs.groupby('user_id')['time_stamp'].diff().dt.days
    
    avg_days_between_purchases = purchase_logs.groupby('user_id')['diff_days'].mean().reset_index().rename(
        columns={'diff_days': 'u_avg_days_between_purchases'}
    )
    # avg_days_between_purchases['u_avg_days_between_purchases'].fillna(185, inplace=True)
    avg_days_between_purchases['u_avg_days_between_purchases'] = avg_days_between_purchases['u_avg_days_between_purchases'].fillna(185)

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        avg_days_between_purchases,
        on='user_id', how='left'
    )
    
    # ------------------------
    # 点击/加入购物车/收藏/购买等比率
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
    
    # 替换无穷大和NaN
    matrix.train_test_matrix[['u_purchase_click_rate','u_cart_click_rate','u_fav_click_rate']] = (
        matrix.train_test_matrix[['u_purchase_click_rate','u_cart_click_rate','u_fav_click_rate']]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    
    # 购物车到购买比率
    matrix.train_test_matrix['u_purchase_cart_rate'] = (
        matrix.train_test_matrix['u_purchase'] / matrix.train_test_matrix['u_cart']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # ------------------------
    # 活跃天数 & 购买天数
    # ------------------------
    user_active_days = user_log.groupby('user_id')['date'].nunique().reset_index().rename(
        columns={'date': 'u_active_days'}
    )
    matrix.train_test_matrix = matrix.train_test_matrix.merge(user_active_days, on='user_id', how='left')
    matrix.train_test_matrix['u_active_days'] = matrix.train_test_matrix['u_active_days'].fillna(0).astype(int)
    
    user_purchase_days = purchase_logs.groupby('user_id')['time_stamp'].nunique().reset_index().rename(
        columns={'time_stamp': 'u_purchase_days'}
    )
    matrix.train_test_matrix = matrix.train_test_matrix.merge(user_purchase_days, on='user_id', how='left')
    matrix.train_test_matrix['u_purchase_days'] = matrix.train_test_matrix['u_purchase_days'].fillna(0).astype(int)
    
    # 列出需要截断和分箱的特征
    features_to_cap = [
        'u_click','u_purchase','u_cart','u_fav','u_iid','u_cid','u_bid','u_mid',
        'u_active_days','u_purchase_days','u_avg_days_between_purchases',
        'u_purchase_click_rate','u_cart_click_rate','u_fav_click_rate','u_purchase_cart_rate'
    ]
    
    # 先按 99% 分位截断
    for col in features_to_cap:
        if col in matrix.train_test_matrix.columns:
            matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)
    
    # 然后对所有需要分箱的数值特征进行分箱
    # 选择需要分箱的特征（根据业务理解和数据分布选择）
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
    为用户特征添加 11/11 当天的交互特征，包括购买次数及其占比，并进行截断和分箱处理。
    """
    user_log = origin_data.user_log_format1.copy()

    # 确保 time_stamp 为 datetime 类型
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    # 标记 11/11
    user_log['is_1111'] = (user_log['time_stamp'].dt.month == 11) & (user_log['time_stamp'].dt.day == 11)

    # 1. 统计每个用户在 11/11 的各类行为次数
    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']
    action_1111 = user_log[user_log['is_1111']].groupby(['user_id', 'action_type']).size().unstack(fill_value=0).reset_index()
    action_1111.columns = ['user_id'] + [f'u_{action}_1111' for action in actions]

    # 2. 统计每个用户的总各类行为次数
    action_total = user_log.groupby(['user_id', 'action_type']).size().unstack(fill_value=0).reset_index()
    action_total.columns = ['user_id'] + [f'u_{action}_total' for action in actions]

    # 3. 合并 11/11 行为次数与总行为次数
    user_1111 = action_1111.merge(action_total, on='user_id', how='left')

    # 4. 填充缺失值为 0
    user_1111.fillna(0, inplace=True)

    # 5. 计算 11/11 行为占比
    for action in actions:
        purchase_1111_col = f'u_{action}_1111'
        purchase_total_col = f'u_{action}_total'
        ratio_col = f'u_{action}_1111_ratio'
        user_1111[ratio_col] = user_1111[purchase_1111_col] / user_1111[purchase_total_col].replace(0, 1)
    
    # 6. 进行截断处理
    features_to_cap = [f'u_{action}_1111' for action in actions] + [f'u_{action}_1111_ratio' for action in actions]
    for col in features_to_cap:
        if col in user_1111.columns:
            user_1111[col] = _cap_values(user_1111[col], upper_percentile=95)

    # 7. 进行分箱处理
    features_to_bin = [f'u_{action}_1111_ratio' for action in actions]
    for col in features_to_bin:
        if col in user_1111.columns:
            bin_col = f"{col}_bin"
            user_1111[bin_col] = _bin_values(user_1111[col], bins=8)

    # 8. 选择需要合并到 train_test_matrix 的特征
    merge_cols = [f'u_{action}_1111' for action in actions] + \
                 [f'u_{action}_1111_ratio' for action in actions] + \
                 [f'u_{action}_1111_ratio_bin' for action in actions]

    # 9. 合并特征到 train_test_matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        user_1111[['user_id'] + merge_cols],
        on='user_id',
        how='left'
    )

    # 10. 处理合并后可能存在的缺失值
    matrix.train_test_matrix[merge_cols] = matrix.train_test_matrix[merge_cols].fillna(0)