# um_feature.py

import pandas as pd
import numpy as np

from helper import _cap_values, _bin_values

def add_user_merchant_features(matrix, origin_data):
    """
    添加与用户-商家组合相关的特征到训练和测试矩阵中，并进行截断与分箱处理以捕捉非线性关系。
    """
    user_log = origin_data.user_log_format1.copy()
    um_group = user_log.groupby(['user_id', 'merchant_id'])

    # ------------------------
    # 基础统计特征
    # ------------------------
    unique_counts = um_group.agg({
        'item_id': 'nunique',
        'cat_id': 'nunique',
        'brand_id': 'nunique'
    }).rename(columns={
        'item_id': 'um_iid',   # 用户-商家唯一商品ID数量
        'cat_id': 'um_cid',    # 用户-商家唯一品类ID数量
        'brand_id': 'um_bid'   # 用户-商家唯一品牌ID数量
    })

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        unique_counts.reset_index(),
        on=['user_id', 'merchant_id'], how='left'
    )

    # 行为计数
    action_counts = user_log.pivot_table(
        index=['user_id', 'merchant_id'],
        columns='action_type',
        aggfunc='size',
        fill_value=0
    ).reset_index().rename(columns={
        'click': 'um_click',            # 每个用户-商家的点击次数
        'add-to-cart': 'um_cart',       # 每个用户-商家的加入购物车次数
        'purchase': 'um_purchase',      # 每个用户-商家的购买次数
        'add-to-favorite': 'um_fav'     # 每个用户-商家的收藏次数
    })

    # 移除不需要的列（这里只移除 'um_cart'，保留 'um_fav' 供后续使用）
    action_counts.drop(['um_cart'], axis=1, inplace=True, errors='ignore')

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        action_counts,
        on=['user_id', 'merchant_id'], how='left'
    )

    # ------------------------
    # 活动持续时间
    # ------------------------
    um_time = um_group['time_stamp'].agg(['min', 'max']).reset_index()
    um_time['um_days_between'] = (um_time['max'] - um_time['min']).dt.days
    um_time['um_days_between'] = um_time['um_days_between'].fillna(0)

    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        um_time[['user_id', 'merchant_id', 'um_days_between']],
        on=['user_id', 'merchant_id'], how='left'
    )

    # 平均每日订单数
    matrix.train_test_matrix['um_avg_orders_per_day'] = (
        matrix.train_test_matrix['um_purchase'] / matrix.train_test_matrix['um_days_between'].replace(0, 1)
    )

    # ------------------------
    # 计算比率特征
    # ------------------------
    matrix.train_test_matrix['um_purchase_per_click'] = (
        matrix.train_test_matrix['um_purchase'] / matrix.train_test_matrix['um_click']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    matrix.train_test_matrix['um_fav_per_click'] = (
        matrix.train_test_matrix['um_fav'] / matrix.train_test_matrix['um_click']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # ------------------------
    # 互动频率特征
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
    # 截断与分箱处理
    # ------------------------
    # 列出需要截断和分箱的特征
    features_to_cap = [
        'um_iid', 'um_cid', 'um_bid',
        'um_click', 'um_purchase',
        'um_days_between', 'um_avg_orders_per_day',
        'um_purchase_per_click', 'um_fav_per_click',
        'um_interaction_freq'
    ]

    # 先按 99% 分位截断
    for col in features_to_cap:
        if col in matrix.train_test_matrix.columns:
            matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col], upper_percentile=95)

    # 然后对所有需要分箱的数值特征进行分箱
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
    计算每个用户与商家的 '加入购物车' 到 '购买' 的平均时间间隔，并将其作为特征添加到训练和测试矩阵中。
    """
    user_log = origin_data.user_log_format1.copy()

    # 仅保留 'add-to-cart' 和 'purchase' 操作
    cart_purchase_log = user_log[user_log['action_type'].isin(['add-to-cart', 'purchase'])].copy()

    # 确保 'time_stamp' 是 datetime 类型
    if cart_purchase_log['time_stamp'].dtype != 'datetime64[ns]':
        cart_purchase_log['time_stamp'] = pd.to_datetime(
            '2016' + cart_purchase_log['time_stamp'].astype(str), format='%Y%m%d', errors='coerce'
        )

    # 排序以确保时间顺序
    cart_purchase_log = cart_purchase_log.sort_values(['user_id', 'merchant_id', 'time_stamp'])

    # 标记是否为购买操作
    cart_purchase_log['is_purchase'] = (cart_purchase_log['action_type'] == 'purchase').astype(int)

    # 使用 shift 找到每次 'add-to-cart' 后的下一个 'purchase' 时间
    cart_purchase_log['next_purchase_time'] = cart_purchase_log.groupby(['user_id', 'merchant_id'])['time_stamp'].shift(-1)
    cart_purchase_log['next_action'] = cart_purchase_log.groupby(['user_id', 'merchant_id'])['action_type'].shift(-1)

    # 仅保留 'add-to-cart' 后紧跟 'purchase' 的记录
    cart_purchase_log = cart_purchase_log[
        (cart_purchase_log['action_type'] == 'add-to-cart') &
        (cart_purchase_log['next_action'] == 'purchase')
    ]

    # 计算时间间隔（天数）
    cart_purchase_log['cart_purchase_interval_days'] = (
        cart_purchase_log['next_purchase_time'] - cart_purchase_log['time_stamp']
    ).dt.days

    # 计算每个用户-商家对的平均时间间隔
    cart_purchase_interval = cart_purchase_log.groupby(['user_id', 'merchant_id'])['cart_purchase_interval_days'].mean().reset_index().rename(
        columns={'cart_purchase_interval_days': 'um_avg_cart_purchase_interval'}
    )

    # 将平均时间间隔合并到训练和测试矩阵中
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        cart_purchase_interval,
        on=['user_id', 'merchant_id'], how='left'
    )

    # 填充缺失值（无对应的 'add-to-cart' -> 'purchase' 操作）
    # matrix.train_test_matrix['um_avg_cart_purchase_interval'].fillna(-1, inplace=True)
    matrix.train_test_matrix['um_avg_cart_purchase_interval'] \
        = matrix.train_test_matrix['um_avg_cart_purchase_interval'].fillna(-1)

    # 对 'um_avg_cart_purchase_interval' 进行截断与分箱
    if 'um_avg_cart_purchase_interval' in matrix.train_test_matrix.columns:
        # 截断
        matrix.train_test_matrix['um_avg_cart_purchase_interval'] = _cap_values(
            matrix.train_test_matrix['um_avg_cart_purchase_interval'], upper_percentile=95
        )
        # 分箱
        matrix.train_test_matrix['um_avg_cart_purchase_interval_bin'] = _bin_values(
            matrix.train_test_matrix['um_avg_cart_purchase_interval'], bins=8
        )