import time
import data_loader
import train_test_matrix
from u_feature import add_user_features
from m_feature import add_merchant_features
from um_feature import add_user_merchant_features
import model

def main():
    print('Start running...\n')

    start = time.time()
    time0 = start

    # Load data
    origin_data = data_loader.DataLoader('../data/')
    origin_data.init_data()

    time1 = time.time()
    print('Load data successfully!')
    print(f'Load data cost {time1 - time0} seconds. Total cost {time1 - start} seconds.\n')
    time0 = time1

    # Generate matrix
    matrix = train_test_matrix.TrainTestMatrix(origin_data)
    matrix.init_matrix()

    time1 = time.time()
    print('Generate matrix successfully!')
    print(f'Generate matrix cost {time1 - time0} seconds. Total cost {time1 - start} seconds.\n')
    time0 = time1

    # Add user, merchant and user merchant pair features
    add_user_features(matrix, origin_data)
    add_merchant_features(matrix, origin_data)
    add_user_merchant_features(matrix, origin_data)
    print(matrix.train_test_matrix.head())

    time1 = time.time()
    print('Add user and merchant features successfully!')
    print(f'Add user and merchant features cost {time1 - time0} seconds. Total cost {time1 - start} seconds.\n')
    time0 = time1

    # Train model
    my_model = model.LightGBMModel(matrix)
    my_model.train_model()

    time1 = time.time()
    print('Train model successfully!')
    print(f'Train model cost {time1 - time0} seconds. Total cost {time1 - start} seconds.\n')
    time0 = time1

    my_model.save_predictions(origin_data, '../submission/submission.csv')

    time1 = time.time()
    print('Save predictions successfully!')
    print(f'Save predictions cost {time1 - time0} seconds. Total cost {time1 - start} seconds.')
    

if __name__ == '__main__':
    main()