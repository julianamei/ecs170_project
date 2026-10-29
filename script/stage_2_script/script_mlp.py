# os builds file paths and folders
# sys lets us change where python looks for code
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# do this to make sure folder is accessed even if not found initially
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader #reads train/test.csv
from local_code.stage_2_code.Method_MLP import Method_MLP #neuralnetwork
from local_code.stage_2_code.Result_Saver import Result_Saver #saves predictions to a file
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test #runs load -> train -> test -> save -> eval
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics #computes accuracy, precision, recall, f1
from local_code.stage_2_code.Plot_Learning_Curve import plot_learning_curve #draws loss/accuracy graph

#needed to set random seed
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2) #fix so every run gives same results
    torch.manual_seed(2) #also fixed
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_2_data') + os.sep # folder with train/test.csv
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result') + os.sep #folder for saving results
    os.makedirs(RESULT_DIR, exist_ok=True)
    EXPERIMENT_NAME = 'baseline' #label so runs don't overwrite
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage 2 data','') #create data loader
    data_obj.dataset_source_folder_path = DATA_DIR #data loader reads from this
    data_obj.dataset_train_file_name = 'train.csv'
    data_obj.dataset_test_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '') #create MLP model
    method_obj.hidden_dims = [256, 128] #2 hidden layers with number of neurons set
    method_obj.activation = 'relu' #activation function after each hidden layer
    method_obj.dropout = 0.2 #randomly turn off this % of neurons during training to reduce overfitting
    method_obj.loss_name = 'cross_entropy' #how prediction errors are measured
    method_obj.optimizer_name = 'adam' #algorithm that updates the weights
    method_obj.learning_rate = 1e-3 #step size for each update
    method_obj.batch_size = 128 #images per weight update
    method_obj.max_epoch = 20 #number of full passes over the training data
    method_obj.device_name = 'auto' #use mac GPU

    result_obj = Result_Saver('saver', '') #create object that saves predictions
    result_obj.result_destination_folder_path = RESULT_DIR + 'MLP_' #save into results folder
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('pre-partitioned train/test', '') #train on train data, test on test data - no random split
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, F1 score', '') #create scorer
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary() #print which dataset/method/setting are being used
    scores = setting_obj.load_run_save_evaluate() #load, train, test, save, score

    # draw loss/accuracy per epoch, save it, show layer sizes/loss/optimizer
    plot_learning_curve(method_obj.history, RESULT_DIR + 'MLP_learning_curve.png', title = 'MLP %s, %s, %s' % (method_obj.hidden_dims, method_obj.loss_name, method_obj.optimizer_name))

    print('************ Overall Performance (testing set) ************')
    for name, value, in scores.items(): #go through every score and print name + value
        print('%-20s %.4f' % (name, value))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    