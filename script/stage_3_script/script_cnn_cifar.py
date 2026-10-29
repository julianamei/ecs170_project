# os builds file paths and folders
# sys lets us change where python looks for code
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# do this to make sure folder is accessed even if not found initially
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader #reads MNIST, ORL, CIFAR files
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR #neuralnetwork
from local_code.stage_3_code.Result_Saver import Result_Saver #saves predictions to a file
from local_code.stage_3_code.Setting_Train_Test import Setting_Train_Test 
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics #computes accuracy, precision, recall, f1
from local_code.stage_3_code.Plot_Learning_Curve import plot_learning_curve #draws loss/accuracy graph

#set random seed
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2) #fix so every run gives same results
    torch.manual_seed(2) #also fixed
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_3_data') + os.sep # folder with the MNIST, ORL, and CIFAR files
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result') + os.sep #folder for saving results
    os.makedirs(RESULT_DIR, exist_ok=True)
    EXPERIMENT_NAME = 'baseline' #label so runs don't overwrite
    #------------------------------------------------------

    # ---- object initialization section ---------------
    data_obj = Dataset_Loader('cifar', '')
    data_obj.dataset_source_folder_path = DATA_DIR
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN_CIFAR('cnn cifar', '')
    method_obj.conv_channels = [32, 64, 128]   # one entry per conv layer
    method_obj.kernel_size = 3
    method_obj.padding = 1
    method_obj.stride = 1
    method_obj.pool = 'max'               # 'max', 'avg', or None
    method_obj.pool_size = 2
    method_obj.hidden_dims = [256]        # fully connected layers after the conv layers
    method_obj.activation = 'relu'
    method_obj.dropout = 0.3
    method_obj.loss_name = 'cross_entropy'
    method_obj.optimizer_name = 'adam'
    method_obj.learning_rate = 1e-3
    method_obj.batch_size = 128
    method_obj.max_epoch = 15              
    method_obj.device_name = 'auto'

    result_obj = Result_Saver('saver', '') #create object that saves predictions
    result_obj.result_destination_folder_path = RESULT_DIR + 'CNN_' #save into results folder
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
    plot_learning_curve(method_obj.history, RESULT_DIR + 'CNN_learning_curve.png', title = 'CNN %s, %s, %s' % (method_obj.conv_channels, method_obj.loss_name, method_obj.optimizer_name))

    print('************ Overall Performance (testing set) ************')
    for name, value, in scores.items(): #go through every score and print name + value
        print('%-20s %.4f' % (name, value))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    