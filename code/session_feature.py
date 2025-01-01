# session_feature.py

import pandas as pd

def add_session_features(matrix, origin_data, session_gap=7):
    user_log = origin_data.user_log_format1.copy()
    
    # 1) Ensure time_stamp is of datetime type
    if user_log['time_stamp'].dtype != 'datetime64[ns]':
        user_log['time_stamp'] = pd.to_datetime(
            '2016' + user_log['time_stamp'], format='%Y%m%d', errors='coerce'
        )
    
    # 2) Sort by user
    user_log = user_log.sort_values(['user_id', 'time_stamp'])
    
    # 3) Calculate session ID
    user_log['prev_time'] = user_log.groupby('user_id')['time_stamp'].shift()
    
    # 4) Calculate time difference, using days as the unit (changed from minutes)
    user_log['gap'] = (user_log['time_stamp'] - user_log['prev_time']).dt.total_seconds() / (60 * 60 * 24)
    
    # 5) Calculate session ID, a new session is considered when the gap is greater than session_gap days
    user_log['session_id'] = (user_log['gap'] > session_gap).cumsum()
    
    # 6) Calculate session-level features
    session_stats = user_log.groupby(['user_id', 'session_id']).agg({
        'action_type': 'count',
        'time_stamp': ['min', 'max']
    })
    session_stats.columns = ['session_action_count', 'session_start', 'session_end']
    session_stats.reset_index(inplace=True)
    session_stats['session_duration'] = (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds()
    
    # 7) Aggregate to user level
    user_session_stats = session_stats.groupby('user_id').agg({
        'session_action_count': ['mean', 'max'],
        'session_duration': ['mean', 'max']
    })
    user_session_stats.columns = [
        'session_action_mean', 'session_action_max',
        'session_duration_mean', 'session_duration_max'
    ]
    user_session_stats.reset_index(inplace=True)
    
    # 8) Merge features into matrix.train_test_matrix
    matrix.train_test_matrix = matrix.train_test_matrix.merge(
        user_session_stats, on='user_id', how='left'
    )
    
    # 9) Fill missing values selectively (only for newly added numerical feature columns)
    new_feature_cols = ['session_action_mean', 'session_action_max', 'session_duration_mean', 'session_duration_max']
    matrix.train_test_matrix[new_feature_cols] = matrix.train_test_matrix[new_feature_cols].fillna(-1)