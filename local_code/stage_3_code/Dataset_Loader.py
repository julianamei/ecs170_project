'''
Concrete IO class for the stage 3 image datasets (MNIST, ORL, CIFAR)
'''

from local_code.base_class.dataset import dataset
import numpy as np
import pickle


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None   # 'MNIST', 'ORL', or 'CIFAR'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _convert(self, instances):
        X = np.array([inst['image'] for inst in instances], dtype=np.float32)
        y = np.array([inst['label'] for inst in instances], dtype=np.int64)

        if self.dataset_source_file_name == 'MNIST':
            X = X[:, None, :, :]                 # (N,28,28) -> (N,1,28,28)
        elif self.dataset_source_file_name == 'ORL':
            X = X[:, :, :, 0][:, None, :, :]     # keep one channel -> (N,1,112,92)
            y = y - 1                            # labels 1-40 -> 0-39
        elif self.dataset_source_file_name == 'CIFAR':
            X = X.transpose(0, 3, 1, 2)          # (N,32,32,3) -> (N,3,32,32)

        return X / 255.0, y

    def load(self):
        print('loading data...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            raw = pickle.load(f)
        X_train, y_train = self._convert(raw['train'])
        X_test, y_test = self._convert(raw['test'])
        return {'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}}