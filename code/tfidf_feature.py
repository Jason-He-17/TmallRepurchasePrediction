# tfidf_feature.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from helper import _cap_values, _bin_values
from helper import remove_original_columns_if_binned

def add_tfidf_features(matrix, origin_data, top_n=64):
    user_log = origin_data.user_log_format1.copy()

    # 确保 time_stamp 为 datetime
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime('2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce')

    # 一、用户级别TF-IDF
    user_log['text_user'] = (user_log['cat_id'].astype(str) + ' ' +
                             user_log['brand_id'].astype(str) + ' ' +
                             user_log['merchant_id'].astype(str) + ' ' +
                             user_log['item_id'].astype(str))
    user_text = user_log.groupby('user_id')['text_user'].apply(lambda x: ' '.join(x)).reset_index()

    vect_user = TfidfVectorizer(max_features=top_n)
    user_matrix = vect_user.fit_transform(user_text['text_user'])
    user_feature_df = pd.DataFrame(user_matrix.toarray(),
                                   columns=[f'u_tfidf_{i}' for i in range(user_matrix.shape[1])])
    user_feature_df['user_id'] = user_text['user_id']

    # 截断和分箱
    tfidf_cols_user = [col for col in user_feature_df.columns if 'tfidf' in col]
    for col in tfidf_cols_user:
        # user_feature_df[col] = _cap_values(user_feature_df[col], upper_percentile=95)
        user_feature_df[f"{col}_bin"] = _bin_values(user_feature_df[col], bins=16)

    # 合并到 matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(user_feature_df, on='user_id', how='left')

    print("用户级别TF-IDF特征已添加并处理。")

    # 二、商家级别TF-IDF
    user_log['text_merchant'] = (user_log['cat_id'].astype(str) + ' ' +
                                 user_log['brand_id'].astype(str) + ' ' +
                                 user_log['item_id'].astype(str))
    merchant_text = user_log.groupby('merchant_id')['text_merchant'].apply(lambda x: ' '.join(x)).reset_index()

    vect_merchant = TfidfVectorizer(max_features=top_n)
    merchant_matrix = vect_merchant.fit_transform(merchant_text['text_merchant'])
    merchant_feature_df = pd.DataFrame(merchant_matrix.toarray(),
                                       columns=[f'm_tfidf_{i}' for i in range(merchant_matrix.shape[1])])
    merchant_feature_df['merchant_id'] = merchant_text['merchant_id']

    # 截断和分箱
    tfidf_cols_merchant = [col for col in merchant_feature_df.columns if 'tfidf' in col]
    for col in tfidf_cols_merchant:
        # merchant_feature_df[col] = _cap_values(merchant_feature_df[col], upper_percentile=95)
        merchant_feature_df[f"{col}_bin"] = _bin_values(merchant_feature_df[col], bins=16)

    # 合并到 matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(merchant_feature_df, on='merchant_id', how='left')

    print("商家级别TF-IDF特征已添加并处理。")

    # 三、用户-商家级别TF-IDF
    user_log['um_text'] = (user_log['cat_id'].astype(str) + ' ' +
                           user_log['brand_id'].astype(str) + ' ' +
                           user_log['item_id'].astype(str))
    um_text = user_log.groupby(['user_id','merchant_id'])['um_text'].apply(lambda x: ' '.join(x)).reset_index()

    vect_um = TfidfVectorizer(max_features=top_n)
    um_matrix = vect_um.fit_transform(um_text['um_text'])
    um_feature_df = pd.DataFrame(um_matrix.toarray(),
                                 columns=[f'um_tfidf_{i}' for i in range(um_matrix.shape[1])])
    um_feature_df['user_id'] = um_text['user_id']
    um_feature_df['merchant_id'] = um_text['merchant_id']

    # 截断和分箱
    tfidf_cols_um = [col for col in um_feature_df.columns if 'tfidf' in col]
    for col in tfidf_cols_um:
        # um_feature_df[col] = _cap_values(um_feature_df[col], upper_percentile=95)
        um_feature_df[f"{col}_bin"] = _bin_values(um_feature_df[col], bins=16)

    # 合并到 matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(um_feature_df, on=['user_id','merchant_id'], how='left')

    print("用户-商家级别TF-IDF特征已添加并处理。")

    # 四、缺失值填充
    tfidf_bin_cols = [col for col in matrix.train_test_matrix.columns if 'tfidf_bin' in col]
    matrix.train_test_matrix[tfidf_bin_cols] = matrix.train_test_matrix[tfidf_bin_cols].fillna(0)

    print("TF-IDF分箱特征缺失值已填充。")