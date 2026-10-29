'''
Stage 2 Result_Saver: pickles the predictions (pred_y / true_y) to disk.
'''

from local_code.base_class.result import result
import pickle


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        path = self.result_destination_folder_path + self.result_destination_file_name
        if self.fold_count is not None:
            path += '_' + str(self.fold_count)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)