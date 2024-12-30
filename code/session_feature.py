# session_feature.py

import pandas as pd

def add_session_features(matrix, origin_data, session_gap=7):
    user_log = origin_data.user_log_format1.copy()
    
    # 1) 确保 time_stamp 为 datetime 类型
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime(
            '2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce'
        )
    
    # 2) 按用户排序
    user_log = user_log.sort_values(['user_id', 'time_stamp'])
    
    # 3) 计算会话ID
    user_log['prev_time'] = user_log.groupby('user_id')['time_stamp'].shift()
    
    # 4) 计算时间差，使用分钟作为单位
    user_log['gap'] = (user_log['time_stamp'] - user_log['prev_time']).dt.total_seconds() / (60 * 60 * 24)
    
    # 5) 计算会话ID，当时间差大于 session_gap 时，认为是新会话
    user_log['session_id'] = (user_log['gap'] > session_gap).cumsum()
    
    # 6) 计算会话级特征
    session_stats = user_log.groupby(['user_id', 'session_id']).agg({
        'action_type': 'count',
        'time_stamp': ['min', 'max']
    })
    session_stats.columns = ['session_action_count', 'session_start', 'session_end']
    session_stats.reset_index(inplace=True)
    session_stats['session_duration'] = (session_stats['session_end'] - session_stats['session_start']).dt.seconds
    
    # 7) 聚合到用户级别
    user_session_stats = session_stats.groupby('user_id').agg({
        'session_action_count': ['mean', 'max'],
        'session_duration': ['mean', 'max']
    })
    user_session_stats.columns = [
        'session_action_mean', 'session_action_max',
        'session_duration_mean', 'session_duration_max'
    ]
    user_session_stats.reset_index(inplace=True)
    
    # 8) 合并特征到 matrix.train_test_matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        user_session_stats, on='user_id', how='left'
    )
    
    # 9) 有选择地填充缺失值（仅针对新加入的数值特征列）
    new_feature_cols = ['session_action_mean', 'session_action_max', 'session_duration_mean', 'session_duration_max']
    matrix.train_test_matrix[new_feature_cols] = matrix.train_test_matrix[new_feature_cols].fillna(-1)