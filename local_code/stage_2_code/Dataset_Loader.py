'''
Stage 2 Dataset_Loader
Loads the pre-partitioned train.csv / test.csv files.
Each line: label,feature1,...,feature784 (all integers, label in 0..9)
'''

from local_code.base_class.dataset import dataset
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = 'train.csv'
    dataset_test_file_name = 'test.csv'
    normalize = True  # scale pixel features from 0..255 to 0..1

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_file(self, file_name):
        path = self.dataset_source_folder_path + file_name
        try:
            import pandas as pd  # much faster for large CSVs
            arr = pd.read_csv(path, header=None).values
        except ImportError:
            arr = np.loadtxt(path, delimiter=',', dtype=np.int64)
        y = arr[:, 0].astype(np.int64)
        X = arr[:, 1:].astype(np.float32)
        if self.normalize:
            X = X / 255.0
        return X, y

    def load(self):
        print('loading data...')
        X_train, y_train = self.load_file(self.dataset_train_file_name)
        X_test, y_test = self.load_file(self.dataset_test_file_name)
        print('train:', X_train.shape, ' test:', X_test.shape,
              ' classes:', sorted(set(y_train.tolist())))
        return {'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}}