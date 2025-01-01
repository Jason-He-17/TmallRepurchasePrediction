# model_ensemble.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

from helper import _cap_values, _bin_values

def add_cross_features(matrix, origin_data):
    """
    Generate cross-feature interactions and merge them into the training and testing matrices.
    For example, combine user age, gender with merchant category, brand information to generate cross-features.
    """
    df = matrix.train_test_matrix.copy()
    
    # Ensure 'age_range' and 'gender' are numeric
    df['age_range'] = pd.to_numeric(df['age_range'], errors='coerce')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    
    # Fill missing values with -1 and convert to integer type
    df['age_range'] = df['age_range'].fillna(-1).astype(int)
    df['gender'] = df['gender'].fillna(-1).astype(int)
    
    # Generate cross-feature for age and gender
    df['age_gender_cross'] = (df['age_range'] * 10) + df['gender']
    
    # Ensure 'm_cid' and 'm_bid' are numeric
    df['m_cid'] = pd.to_numeric(df['m_cid'], errors='coerce').fillna(-1).astype(int)
    df['m_bid'] = pd.to_numeric(df['m_bid'], errors='coerce').fillna(-1).astype(int)
    
    # Generate cross-feature for category and brand
    df['cat_brand_cross'] = (df['m_cid'] * 10000) + df['m_bid']
    
    # Assign generated cross-features directly to train_test_matrix to avoid using merge
    matrix.train_test_matrix['age_gender_cross'] = df['age_gender_cross']
    matrix.train_test_matrix['cat_brand_cross'] = df['cat_brand_cross']
    
    # Fill missing values
    matrix.train_test_matrix['age_gender_cross'] = matrix.train_test_matrix['age_gender_cross'].fillna(0).astype(int)
    matrix.train_test_matrix['cat_brand_cross'] = matrix.train_test_matrix['cat_brand_cross'].fillna(0).astype(int)
    
    # Capping treatment
    matrix.train_test_matrix['age_gender_cross'] = _cap_values(
        matrix.train_test_matrix['age_gender_cross'], upper_percentile=95
    )
    matrix.train_test_matrix['cat_brand_cross'] = _cap_values(
        matrix.train_test_matrix['cat_brand_cross'], upper_percentile=95
    )
    
    # Binning treatment
    matrix.train_test_matrix['age_gender_cross_bin'] = _bin_values(
        matrix.train_test_matrix['age_gender_cross'], bins=8
    )
    matrix.train_test_matrix['cat_brand_cross_bin'] = _bin_values(
        matrix.train_test_matrix['cat_brand_cross'], bins=8
    )

def add_model_ensemble_features(matrix, origin_data, n_splits=5, random_state=42):
    """
    Use the predicted probabilities from Random Forest and XGBoost models as new features to enhance the main model's performance.
    
    :param matrix: Training/testing data matrix object containing the train_test_matrix attribute
    :param origin_data: Original data object containing the user_log_format1 attribute
    :param n_splits: Number of folds for K-fold cross-validation
    :param random_state: Seed for randomness
    """
    df = matrix.train_test_matrix.copy()
    train_df = df[df['origin'] == 'train'].reset_index(drop=True)
    test_df = df[df['origin'] == 'test'].reset_index(drop=True)
    
    # Features and labels
    feature_cols = [col for col in train_df.columns if col not in ['user_id', 'merchant_id', 'label', 'origin']]
    X = train_df[feature_cols]
    y = train_df['label']
    
    X_test = test_df[feature_cols]
    
    # Encode non-numeric features
    # Use LabelEncoder for all non-numeric features
    # If there are categories that cannot be encoded, they will be assigned a new label
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            # If categorical type, first convert to string type to allow adding 'missing'
            if X[col].dtype.name == 'category':
                X[col] = X[col].astype('object')
                X_test[col] = X_test[col].astype('object')
            
            le = LabelEncoder()
            # Fill missing values with 'missing' before LabelEncoding to prevent LabelEncoder error
            X[col] = X[col].fillna('missing').astype(str)
            X_test[col] = X_test[col].fillna('missing').astype(str)
            # Concatenate train and test sets for encoding to ensure categories in the test set are also encoded
            combined = pd.concat([X[col], X_test[col]], axis=0)
            le.fit(combined)
            X[col] = le.transform(X[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le  # Save encoders for later use
    
    # Ensure all features are numeric
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Non-encoded non-numeric features exist: {non_numeric_cols.tolist()}")
    
    # Initialize prediction probability columns
    rf_preds = np.zeros(len(train_df))
    xgb_preds = np.zeros(len(train_df))
    
    rf_test_preds = np.zeros(len(test_df))
    xgb_test_preds = np.zeros(len(test_df))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Processing fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds[val_index] = rf.predict_proba(X_val)[:, 1]
        rf_test_preds += rf.predict_proba(X_test)[:, 1] / n_splits
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='auc'
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds[val_index] = xgb_model.predict_proba(X_val)[:, 1]
        xgb_test_preds += xgb_model.predict_proba(X_test)[:, 1] / n_splits
    
    # Add prediction probabilities as new features
    train_rf_pred_col = pd.Series(rf_preds, index=train_df.index, name='rf_pred')
    test_rf_pred_col = pd.Series(rf_test_preds, index=test_df.index, name='rf_pred')
    train_xgb_pred_col = pd.Series(xgb_preds, index=train_df.index, name='xgb_pred')
    test_xgb_pred_col = pd.Series(xgb_test_preds, index=test_df.index, name='xgb_pred')
    
    # Assign new features back to the original matrix
    matrix.train_test_matrix.loc[train_df.index, 'rf_pred'] = train_rf_pred_col
    matrix.train_test_matrix.loc[test_df.index, 'rf_pred'] = test_rf_pred_col
    
    matrix.train_test_matrix.loc[train_df.index, 'xgb_pred'] = train_xgb_pred_col
    matrix.train_test_matrix.loc[test_df.index, 'xgb_pred'] = test_xgb_pred_col
    
    # Fill missing values
    matrix.train_test_matrix['rf_pred'] = matrix.train_test_matrix['rf_pred'].fillna(0)
    matrix.train_test_matrix['xgb_pred'] = matrix.train_test_matrix['xgb_pred'].fillna(0)
    
    # Capping treatment
    matrix.train_test_matrix['rf_pred'] = _cap_values(
        matrix.train_test_matrix['rf_pred'], upper_percentile=95
    )
    matrix.train_test_matrix['xgb_pred'] = _cap_values(
        matrix.train_test_matrix['xgb_pred'], upper_percentile=95
    )
    
    # Binning treatment
    matrix.train_test_matrix['rf_pred_bin'] = _bin_values(
        matrix.train_test_matrix['rf_pred'], bins=8
    )
    matrix.train_test_matrix['xgb_pred_bin'] = _bin_values(
        matrix.train_test_matrix['xgb_pred'], bins=8
    )