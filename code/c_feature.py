# c_feature.py

import numpy as np
import pandas as pd

from helper import _cap_values, _bin_values

def add_category_features(matrix, origin_data):
    """
    1. 对全局 cat_id 做聚合统计，得到分类特征的统计信息。
    2. 对 (user_id, merchant_id, cat_id) 分组，选取最常见的 cat_id 记为 top_cat_id。
    3. 合并最常见 cat_id 的统计信息至 (user_id, merchant_id) 级别，再合并到训练矩阵。
    4. 统一做 95% 分位截断 + 8 分箱，并移除无用的 datetime 列。
    """
    user_log = origin_data.user_log_format1.copy()

    # 确保 time_stamp 为 datetime 类型
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        # 若原本是字符串格式，如 '20161111'，则使用 format='%Y%m%d'
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    ##############################
    # 1) 全局分类特征聚合: cat_id #
    ##############################
    cat_group = user_log.groupby('cat_id').agg({
        'user_id': 'nunique',
        'merchant_id': 'nunique',
        'item_id': 'nunique',
        'time_stamp': ['min', 'max'],
        # 统计购买次数
        'action_type': lambda x: (x == 'purchase').sum()
    })
    cat_group.columns = [
        'c_unique_users', 'c_unique_merchants', 'c_unique_items',
        'c_min_time', 'c_max_time', 'c_purchase_count'
    ]
    cat_group.reset_index(inplace=True)

    # 计算持续天数
    cat_group['c_days_between'] = (
        cat_group['c_max_time'] - cat_group['c_min_time']
    ).dt.days.fillna(0)

    # 计算平均购买频次（天为尺度）
    cat_group['c_avg_purchases_per_day'] = cat_group.apply(
        lambda row: row['c_purchase_count'] / row['c_days_between'] if row['c_days_between'] > 0 else 0,
        axis=1
    )

    # 移除 datetime 列，后续不再需要
    cat_group.drop(['c_min_time', 'c_max_time'], axis=1, inplace=True)

    # user_log 里按 cat_id 分别统计各类行为次数
    cat_action_counts = user_log.pivot_table(
        index='cat_id', columns='action_type', aggfunc='size', fill_value=0
    ).reset_index().rename(columns={
        'click': 'c_click',
        'add-to-cart': 'c_cart',
        'purchase': 'c_purchase',
        'add-to-favorite': 'c_fav'
    })
    # 和 cat_group 合并
    cat_group = cat_group.merge(cat_action_counts, on='cat_id', how='left')

    # 计算比率
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
    # 2) user_id, merchant_id 中找最常见 cat_id => top_cat_id
    ###############################
    umc = user_log.groupby(['user_id', 'merchant_id', 'cat_id']).size().reset_index(name='freq_cat')
    # 按 freq_cat 降序，然后对 (user_id, merchant_id) 去重 => 得到最常见 cat_id
    umc.sort_values('freq_cat', ascending=False, inplace=True)
    umc.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first', inplace=True)
    umc.rename(columns={'cat_id': 'top_cat_id'}, inplace=True)

    # 把 cat_group 拼上
    umc = umc.merge(cat_group, left_on='top_cat_id', right_on='cat_id', how='left').drop('cat_id', axis=1)

    #####################################################
    # 3) 将最常见 top_cat_id 的统计信息拼回训练矩阵
    #####################################################
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        umc, on=['user_id', 'merchant_id'], how='left'
    )

    ###################################
    # 4) 统一做 95% 分位截断 + 8 分箱
    ###################################
    # 挑出要截断的列（以 c_ 开头的数值列都需要截断）
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

    # 挑出要分箱的列（根据其他文件的设置，使用 8 分箱）
    features_to_bin = [
        'c_days_between', 'c_avg_purchases_per_day',
        'c_purchase_click_rate', 'c_cart_click_rate', 'c_fav_click_rate'
    ]
    for col in features_to_bin:
        if col in matrix.train_test_matrix.columns:
            bin_col = f'{col}_bin'
            matrix.train_test_matrix[bin_col] = _bin_values(matrix.train_test_matrix[col], bins=8)

    # 最终完成，无需移除 datetime 列，因为已在步骤 1 移除

def add_category_1111_features(matrix, origin_data):
    """
    针对分类做 11/11 特征，需先找 (user_id, merchant_id) 最常见 cat_id，然后把 cat_1111 特征合并上来。
    """
    user_log = origin_data.user_log_format1.copy()
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    user_log['is_1111'] = (user_log['time_stamp'].dt.month == 11) & (user_log['time_stamp'].dt.day == 11)
    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']

    # 1) 统计 cat 在 11/11 的行为
    c_1111 = user_log[user_log['is_1111']].groupby(['cat_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in c_1111.columns:
            c_1111[act] = 0
    c_1111.columns = [f'c_{a}_1111' for a in actions]
    c_1111.reset_index(inplace=True)

    # 2) 统计 cat 总行为
    c_total = user_log.groupby(['cat_id', 'action_type']).size().unstack(fill_value=0)
    for act in actions:
        if act not in c_total.columns:
            c_total[act] = 0
    c_total.columns = [f'c_{a}_total' for a in actions]
    c_total.reset_index(inplace=True)

    # 3) 合并并计算占比
    cat_1111 = c_1111.merge(c_total, on='cat_id', how='left').fillna(0)
    for act in actions:
        cat_1111[f'c_{act}_1111_ratio'] = (
            cat_1111[f'c_{act}_1111'] / cat_1111[f'c_{act}_total'].replace(0, 1)
        )

    # 4) 截断 & 分箱
    for col in cat_1111.columns:
        if col.startswith('c_') and col != 'cat_id':
            cat_1111[col] = _cap_values(cat_1111[col], upper_percentile=95)

    for act in actions:
        ratio_col = f'c_{act}_1111_ratio'
        if ratio_col in cat_1111.columns:
            bin_col = f'{ratio_col}_bin'
            cat_1111[bin_col] = _bin_values(cat_1111[ratio_col], bins=8)

    # 5) 找最常见 cat_id => top_cat_id
    umc = user_log.groupby(['user_id', 'merchant_id', 'cat_id']).size().reset_index(name='freq_cat')
    umc.sort_values('freq_cat', ascending=False, inplace=True)
    umc.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first', inplace=True)
    umc.rename(columns={'cat_id': 'top_cat_id'}, inplace=True)

    # 6) 拼上 1111 特征
    umc = umc.merge(cat_1111, left_on='top_cat_id', right_on='cat_id', how='left').drop('cat_id', axis=1)

    # 7) 合并进 matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        umc, on=['user_id', 'merchant_id'], how='left'
    )

    ###################################
    # 8) 仅填充数值型列的缺失值
    ###################################
    numerical_cols = cat_1111.columns.drop(['cat_id'])
    numerical_cols = [col for col in numerical_cols if col in matrix.train_test_matrix.columns]
    matrix.train_test_matrix[numerical_cols] = matrix.train_test_matrix[numerical_cols].fillna(0)