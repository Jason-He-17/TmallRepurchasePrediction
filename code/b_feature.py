# b_feature.py

import numpy as np
import pandas as pd

from helper import _cap_values, _bin_values

def add_brand_features(matrix, origin_data):
    """
    1. 对全局 brand_id 做聚合统计，得到品牌特征的统计信息。
    2. 对 (user_id, merchant_id, brand_id) 分组，选取最常见的 brand_id 记为 top_brand_id。
    3. 合并最常见 brand_id 的统计信息至 (user_id, merchant_id) 级别，再合并到训练矩阵。
    4. 统一做 95% 分位截断 + 8 分箱，并移除无用的 datetime 列。
    """
    user_log = origin_data.user_log_format1.copy()

    # 确保 time_stamp 为 datetime 类型
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        # 若原本是字符串格式，如 '20161111'，则使用 format='%Y%m%d'
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    #################################
    # 1) 全局品牌特征聚合: brand_id  #
    #################################
    brand_group = user_log.groupby('brand_id').agg({
        'merchant_id': 'nunique',
        'item_id': 'nunique',
        'time_stamp': ['min', 'max'],
        # 统计购买次数
        'action_type': lambda x: (x == 'purchase').sum()
    })
    brand_group.columns = [
        'b_unique_merchants', 'b_unique_items',
        'b_min_time', 'b_max_time', 'b_purchase_count'
    ]
    brand_group.reset_index(inplace=True)

    # 计算持续天数
    brand_group['b_days_between'] = (
        brand_group['b_max_time'] - brand_group['b_min_time']
    ).dt.days.fillna(0)

    # 计算平均购买频次（天为尺度）
    brand_group['b_avg_purchases_per_day'] = brand_group.apply(
        lambda row: row['b_purchase_count'] / row['b_days_between'] if row['b_days_between'] > 0 else 0,
        axis=1
    )

    # 移除 datetime 列，后续不再需要
    brand_group.drop(['b_min_time', 'b_max_time'], axis=1, inplace=True)

    # user_log 里按 brand_id 分别统计各类行为次数
    brand_action_counts = user_log.pivot_table(
        index='brand_id', columns='action_type', aggfunc='size', fill_value=0
    ).reset_index().rename(columns={
        'click': 'b_click',
        'add-to-cart': 'b_cart',
        'purchase': 'b_purchase',
        'add-to-favorite': 'b_fav'
    })
    # 和 brand_group 合并
    brand_group = brand_group.merge(brand_action_counts, on='brand_id', how='left')

    # 计算比率
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
    # 2) user_id, merchant_id 中找最常见 brand_id => top_brand_id
    ###################################
    umb = user_log.groupby(['user_id', 'merchant_id', 'brand_id']).size().reset_index(name='freq_brand')
    # 按 freq_brand 降序，然后对 (user_id, merchant_id) 去重 => 得到最常见 brand_id
    umb.sort_values('freq_brand', ascending=False, inplace=True)
    umb.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first', inplace=True)
    umb.rename(columns={'brand_id': 'top_brand_id'}, inplace=True)

    # 把 brand_group 拼上
    umb = umb.merge(brand_group, left_on='top_brand_id', right_on='brand_id', how='left').drop('brand_id', axis=1)

    #####################################################
    # 3) 将最常见 top_brand_id 的统计信息拼回训练矩阵
    #####################################################
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        umb, on=['user_id', 'merchant_id'], how='left'
    )

    ###################################
    # 4) 统一做 95% 分位截断 + 8 分箱
    ###################################
    # 挑出要截断的列（以 b_ 开头的数值列都需要截断）
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

    # 挑出要分箱的列（根据其他文件的设置，使用 8 分箱）
    features_to_bin = [
        'b_days_between', 'b_avg_purchases_per_day',
        'b_purchase_click_rate', 'b_cart_click_rate', 'b_fav_click_rate'
    ]
    for col in features_to_bin:
        if col in matrix.train_test_matrix.columns:
            bin_col = f'{col}_bin'
            matrix.train_test_matrix[bin_col] = _bin_values(matrix.train_test_matrix[col], bins=8)

    # 最终完成，无需移除 datetime 列，因为已在步骤 1 移除

def add_brand_1111_features(matrix, origin_data):
    """
    针对品牌做 11/11 特征，需先找 (user_id, merchant_id) 最常见 brand_id，然后把 brand_1111 特征合并上来。
    """
    user_log = origin_data.user_log_format1.copy()
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    user_log['is_1111'] = (user_log['time_stamp'].dt.month == 11) & (user_log['time_stamp'].dt.day == 11)
    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']

    # 1) 统计 brand 在 11/11 的行为
    b_1111 = user_log[user_log['is_1111']].groupby(['brand_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in b_1111.columns:
            b_1111[act] = 0
    b_1111.columns = [f'b_{a}_1111' for a in actions]
    b_1111.reset_index(inplace=True)

    # 2) 统计 brand 总行为
    b_total = user_log.groupby(['brand_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in b_total.columns:
            b_total[act] = 0
    b_total.columns = [f'b_{a}_total' for a in actions]
    b_total.reset_index(inplace=True)

    # 3) 合并并计算占比
    brand_1111 = b_1111.merge(b_total, on='brand_id', how='left').fillna(0)
    for act in actions:
        brand_1111[f'b_{act}_1111_ratio'] = (
            brand_1111[f'b_{act}_1111'] / brand_1111[f'b_{act}_total'].replace(0, 1)
        )

    # 4) 截断 & 分箱
    for col in brand_1111.columns:
        if col.startswith('b_') and col != 'brand_id':
            brand_1111[col] = _cap_values(brand_1111[col], upper_percentile=95)

    for act in actions:
        ratio_col = f'b_{act}_1111_ratio'
        if ratio_col in brand_1111.columns:
            bin_col = f'{ratio_col}_bin'
            brand_1111[bin_col] = _bin_values(brand_1111[ratio_col], bins=8)

    # 5) 找最常见 brand_id => top_brand_id (和 add_brand_features 类似)
    umb = user_log.groupby(['user_id', 'merchant_id', 'brand_id']).size().reset_index(name='freq_brand')
    umb.sort_values('freq_brand', ascending=False, inplace=True)
    umb.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first', inplace=True)
    umb.rename(columns={'brand_id': 'top_brand_id'}, inplace=True)

    # 6) 拼上 1111 特征
    umb = umb.merge(brand_1111, left_on='top_brand_id', right_on='brand_id', how='left').drop('brand_id', axis=1)

    # 7) 合并进 matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        umb, on=['user_id', 'merchant_id'], how='left'
    )

    ###################################
    # 8) 仅填充数值型列的缺失值
    ###################################
    numerical_cols = brand_1111.columns.drop(['brand_id'])
    numerical_cols = [col for col in numerical_cols if col in matrix.train_test_matrix.columns]
    matrix.train_test_matrix[numerical_cols] = matrix.train_test_matrix[numerical_cols].fillna(0)