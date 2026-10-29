'''
Stage 2 setting: the data is already split into train.csv and test.csv,
so there is no random split and no cross validation here.
Train on the training set once, test on the testing set once.
'''

from local_code.base_class.setting import setting


class Setting_Train_Test(setting):

    def load_run_save_evaluate(self):
        # load the pre-partitioned dataset
        loaded_data = self.dataset.load()

        # run MethodModule
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        # evaluate on the testing set
        self.evaluate.data = learned_result
        return self.evaluate.evaluate()