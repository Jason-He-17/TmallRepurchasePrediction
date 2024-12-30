# model.py
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np

class LightGBMModel:
    def __init__(self, matrix):
        self.matrix = matrix
        self.train_data = self._prepare_data(origin='train').drop(['origin'], axis=1)
        self.test_data = self._prepare_data(origin='test').drop(['label', 'origin'], axis=1)
        self.train_x, self.train_y = self.train_data.drop(['label'], axis=1), self.train_data['label']
        self.params = self._set_params()
        self.model = None

    def _prepare_data(self, origin):
        data = self.matrix.train_test_matrix[self.matrix.train_test_matrix['origin'] == origin]
        return data

    def _set_params(self):
        return {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': True,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'early_stopping_rounds': 100
        }

    def k_fold_cross_validation(self, num_boost_round=5000, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_scores = []
        fold = 1

        for train_index, val_index in kf.split(self.train_x):
            X_train, X_val = self.train_x.iloc[train_index], self.train_x.iloc[val_index]
            y_train, y_val = self.train_y.iloc[train_index], self.train_y.iloc[val_index]

            lgb_train = lgb.Dataset(X_train, label=y_train)
            lgb_eval = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

            model = lgb.train(
                self.params,
                lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_train, lgb_eval]
            )

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            auc = roc_auc_score(y_val, y_pred)
            auc_scores.append(auc)
            print(f'Fold {fold} AUC: {auc:.6f}')
            fold += 1

        print(f'Average AUC: {np.mean(auc_scores):.6f}')

    def train(self, num_boost_round=5000):
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.train_x, self.train_y, test_size=0.2, random_state=42
        )
        self.lgb_train = lgb.Dataset(self.x_train, self.y_train)
        self.lgb_eval = lgb.Dataset(self.x_valid, self.y_valid, reference=self.lgb_train)

        self.model = lgb.train(
            self.params,
            self.lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[self.lgb_train, self.lgb_eval]
        )
        return self.model

    def evaluate(self):
        y_pred = self.model.predict(self.x_valid, num_iteration=self.model.best_iteration)
        auc_score = roc_auc_score(self.y_valid, y_pred)
        print(f"LightGBM AUC: {auc_score:.6f}")

    def train_model(self, use_kfold=False, num_boost_round=5000, n_splits=5):
        if use_kfold:
            self.k_fold_cross_validation(num_boost_round=num_boost_round, n_splits=n_splits)
        self.train(num_boost_round=num_boost_round)
        self.evaluate()

    def save_predictions(self, origin_data, output_path):
        submission = origin_data.submission_data_format1.drop(['origin'], axis=1)
        submission['prob'] = self.model.predict(self.test_data, num_iteration=self.model.best_iteration, predict_disable_shape_check=True)
        submission.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")

    def get_booster(self):
        return self.model