import pandas as pd
import train_test_matrix

class TrainTestMatrix:
    def __init__(self, origin_data):
        # Merge data
        self.train_test_matrix = \
            pd.concat([origin_data.train_data_format1, origin_data.submission_data_format1], ignore_index=True, sort=False)
        self.train_test_matrix = self.train_test_matrix.merge(origin_data.user_info_format1, on='user_id', how='left')

    def rename_columns(self):
        # Rename gender column. 0 for female, 1 for male, 2 or NULL for unknown
        self.train_test_matrix['gender'] = self.train_test_matrix['gender'].map({
            0: 'female',
            1: 'male',
            2: 'unknown'
        }).fillna('unknown')
        # Rename age_range column. NULL for unknown
        self.train_test_matrix['age_range'] = self.train_test_matrix['age_range'].map({
            1: 'first group',
            2: 'second group',
            3: 'third group',
            4: 'fourth group',
            5: 'fifth group',
            6: 'sixth group',
            7: 'seventh group',
            8: 'eighth group'
        }).fillna('unknown')

    def change_to_type(self, column, type):
        self.train_test_matrix[column] = self.train_test_matrix[column].astype(type)

    def convert_data_types(self):
        # Convert data types
        self.change_to_type('user_id', 'uint32')
        self.change_to_type('merchant_id', 'uint16')
        self.change_to_type('label', 'float64')
        self.change_to_type('origin', 'category')
        self.change_to_type('age_range', 'category')
        self.change_to_type('gender', 'category')

    def init_matrix(self):
        self.rename_columns()
        self.convert_data_types()