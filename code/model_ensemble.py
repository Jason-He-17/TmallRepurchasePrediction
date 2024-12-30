import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from helper import _cap_values, _bin_values

def add_model_ensemble_features(matrix, origin_data, n_splits=5, random_state=42):
    df = matrix.train_test_matrix.copy()
    train_df = df[df['origin'] == 'train'].reset_index(drop=True)
    test_df = df[df['origin'] == 'test'].reset_index(drop=True)
    
    feature_cols = [col for col in train_df.columns if col not in ['user_id','merchant_id','label','origin']]
    X = train_df[feature_cols]
    y = train_df['label']
    X_test = test_df[feature_cols]
    
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            if X[col].dtype.name == 'category':
                X[col] = X[col].astype('object')
                X_test[col] = X_test[col].astype('object')
            le = LabelEncoder()
            X[col] = X[col].fillna('missing').astype(str)
            X_test[col] = X_test[col].fillna('missing').astype(str)
            combined = pd.concat([X[col], X_test[col]], axis=0)
            le.fit(combined)
            X[col] = le.transform(X[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le

    non_numeric_cols = X.select_dtypes(include=['object','category']).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"存在未编码的非数值特征: {non_numeric_cols.tolist()}")

    # 初始化所有模型预测概率
    rf_preds = np.zeros(len(train_df));  rf_test_preds = np.zeros(len(test_df))
    xgb_preds = np.zeros(len(train_df)); xgb_test_preds = np.zeros(len(test_df))
    lr_preds = np.zeros(len(train_df));  lr_test_preds = np.zeros(len(test_df))
    cat_preds = np.zeros(len(train_df)); cat_test_preds = np.zeros(len(test_df))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Processing fold {fold+1}/{n_splits}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # RandomForest
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds[val_index] = rf.predict_proba(X_val)[:,1]
        rf_test_preds += rf.predict_proba(X_test)[:,1] / n_splits

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=random_state,
            use_label_encoder=False, eval_metric='auc'
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds[val_index] = xgb_model.predict_proba(X_val)[:,1]
        xgb_test_preds += xgb_model.predict_proba(X_test)[:,1] / n_splits

        # Logistic Regression
        lr = LogisticRegression(max_iter=500, random_state=random_state)
        lr.fit(X_train, y_train)
        lr_preds[val_index] = lr.predict_proba(X_val)[:,1]
        lr_test_preds += lr.predict_proba(X_test)[:,1] / n_splits

        # CatBoost
        cat = CatBoostClassifier(
            iterations=100, learning_rate=0.05, depth=6, 
            random_state=random_state, verbose=False
        )
        cat.fit(X_train, y_train)
        cat_preds[val_index] = cat.predict_proba(X_val)[:,1]
        cat_test_preds += cat.predict_proba(X_test)[:,1] / n_splits

    # 添加预测概率到数据矩阵
    train_df['rf_pred'] = rf_preds;   test_df['rf_pred'] = rf_test_preds
    train_df['xgb_pred'] = xgb_preds; test_df['xgb_pred'] = xgb_test_preds
    train_df['lr_pred'] = lr_preds;   test_df['lr_pred'] = lr_test_preds
    train_df['cat_pred'] = cat_preds; test_df['cat_pred'] = cat_test_preds

    # 合并回原矩阵
    for pred_col in ['rf_pred','xgb_pred','lr_pred','cat_pred']:
        matrix.train_test_matrix.loc[train_df.index, pred_col] = train_df[pred_col]
        matrix.train_test_matrix.loc[test_df.index, pred_col] = test_df[pred_col]

    for col in ['rf_pred','xgb_pred','lr_pred','cat_pred']:
        matrix.train_test_matrix[col] = matrix.train_test_matrix[col].fillna(0)
        matrix.train_test_matrix[col] = _cap_values(matrix.train_test_matrix[col],upper_percentile=95)
        matrix.train_test_matrix[f'{col}_bin'] = _bin_values(matrix.train_test_matrix[col],bins=8)