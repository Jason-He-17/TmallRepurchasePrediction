import pandas as pd

class DataLoader:
    def __init__(self, data_path):

        self.user_log_format1 = pd.read_csv(data_path + '/data_format1/user_log_format1.csv', dtype={'time_stamp':'str'})
        self.user_info_format1 = pd.read_csv(data_path + '/data_format1/user_info_format1.csv')
        self.train_data_format1 = pd.read_csv(data_path + '/data_format1/train_format1.csv')
        self.submission_data_format1 = pd.read_csv(data_path + '/data_format1/test_format1.csv')

        self.data_train_format2 = pd.read_csv(data_path + '/data_format2/train_format2.csv')
        self.data_submission_format2 = pd.read_csv(data_path + '/data_format2/test_format2.csv')

    def change_user_log_to_type(self, column, type):
        self.user_log_format1[column] = self.user_log_format1[column].astype(type)
    
    def convert_user_log_data_types(self):
        # Convert data types
        self.change_user_log_to_type('user_id', 'uint32')
        self.change_user_log_to_type('item_id', 'uint32')
        self.change_user_log_to_type('cat_id', 'uint16')
        self.change_user_log_to_type('merchant_id', 'uint16')
        self.change_user_log_to_type('brand_id', 'int16')
        self.user_log_format1['time_stamp'] = pd.to_datetime('2016' + self.user_log_format1['time_stamp'], format='%Y%m%d')
        self.change_user_log_to_type('action_type', 'object')

    def init_data(self):        
        # Tag origin
        self.train_data_format1['origin'] = 'train'
        self.submission_data_format1['origin'] = 'test'
        self.submission_data_format1.drop(['prob'], axis=1, inplace=True)

        # Rename seller_id column to merchant_id
        self.user_log_format1.rename(columns={'seller_id':'merchant_id'}, inplace=True)

        # Rename action_type column. 0 for click, 1 for add-to-cart, 2 for purchase, 3 for add-to-favorite
        self.user_log_format1['action_type'] = self.user_log_format1['action_type'].map({
            0: 'click',
            1: 'add-to-cart',
            2: 'purchase',
            3: 'add-to-favorite'
        })

        # Fill in the missing values of brand_id with 0
        self.user_log_format1['brand_id'] = self.user_log_format1['brand_id'].fillna(0)

        self.convert_user_log_data_types()
