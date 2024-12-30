# mc_feature.py

import numpy as np
import pandas as pd

from helper import _cap_values, _bin_values

def add_merchant_category_features(matrix, origin_data):
    """
    统计 (merchant_id, cat_id) 关系，计算店铺在各类目的交互情况，并合并基于最常见 cat_id 的特征。
    """
    user_log = origin_data.user_log_format1.copy()

    # 确保 time_stamp 为 datetime 类型
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        # 若原本是字符串格式，如 '20161111'，则使用 format='%Y%m%d'
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    actions = ['click', 'add-to-cart', 'purchase', 'add-to-favorite']

    # 1) 按店铺与分类分组，统计多种行为次数
    mc_group = user_log.groupby(['merchant_id', 'cat_id']).agg({
        'user_id': 'nunique',  # 不同用户数
        'item_id': 'nunique',  # 不同商品数
        'action_type': [
            lambda x: (x == 'click').sum(),
            lambda x: (x == 'purchase').sum(),
            lambda x: (x == 'add-to-cart').sum(),
            lambda x: (x == 'add-to-favorite').sum()
        ]
    })

    # 重命名列
    mc_group.columns = [
        'mc_unique_users', 
        'mc_unique_items',
        'mc_click', 
        'mc_purchase', 
        'mc_cart', 
        'mc_fav'
    ]
    mc_group.reset_index(inplace=True)

    # 2) 计算转化率类特征
    mc_group['mc_purchase_click_rate'] = (
        mc_group['mc_purchase'] / mc_group['mc_click'].replace(0, 1)
    )

    # 3) 截断 & 分箱
    features_to_cap = ['mc_click','mc_purchase','mc_cart','mc_fav','mc_unique_users','mc_unique_items','mc_purchase_click_rate']
    for col in features_to_cap:
        if col in mc_group.columns:
            mc_group[col] = _cap_values(mc_group[col], upper_percentile=95)

    for col in ['mc_purchase_click_rate']:
        bin_col = f'{col}_bin'
        if col in mc_group.columns:
            mc_group[bin_col] = _bin_values(mc_group[col], bins=8)

    # 4) 找每个 merchant_id 最常见的 cat_id
    mc_top = user_log.groupby(['merchant_id', 'cat_id']).size().reset_index(name='freq_cat')
    mc_top = mc_top.sort_values(['merchant_id', 'freq_cat'], ascending=[True, False])
    mc_top = mc_top.drop_duplicates(subset=['merchant_id'], keep='first')
    mc_top.rename(columns={'cat_id': 'top_cat_id'}, inplace=True)

    # 5) 合并 mc_group 与 mc_top 以获取每个 merchant_id 的最常见 cat_id 的特征
    mc_features = mc_top.merge(mc_group, left_on=['merchant_id', 'top_cat_id'], right_on=['merchant_id', 'cat_id'], how='left')
    mc_features.drop(['cat_id', 'freq_cat'], axis=1, inplace=True)

    # 6) 合并进 matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        mc_features.drop('top_cat_id', axis=1),
        on='merchant_id',
        how='left'
    )

    # 7) 填充数值类型缺失值
    numerical_cols = mc_features.columns.drop(['merchant_id', 'top_cat_id'])
    numerical_cols = [col for col in numerical_cols if col in matrix.train_test_matrix.columns]
    matrix.train_test_matrix[numerical_cols] = matrix.train_test_matrix[numerical_cols].fillna(0)