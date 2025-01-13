import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping

from niapy.problems import Problem
from niapy.task import Task, OptimizationType
import numpy as np
import pandas as pd
from Alibaba_helper_functions import loadDatasetObj,save_object,MAPE,MAE,RMSE
from Alibaba_helper_functions import expand_dims,list_to_array,get_EN_DE_LSTM_model,get_en_de_lstm_model_attention
from Alibaba_fet_features_LSTM_no_cluster import get_dataset_alibaba_lstm_no_cluster
from niapy.algorithms.basic import CuckooSearch
import warnings
from keras.callbacks import ReduceLROnPlateau

import time
# warnings.filterwarnings('ignore')
# np.random.seed(7)

from Alibaba_helper_functions import get_google_data, get_BB_data
# X_BB.Y_BB = get_BB_data(seq,scaler)
# X_google.Y_google = get_google_data(seq,scaler)
def log_results_LSTM(row,save_path,save_name):
    # save_name = 'results_LSTM_en_de_HP_seach.csv'
    cols = ["RMSE", "MAE", "MAPE(%)","seq","num_layers","units"]
    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    flag = 0
    if len(df)!=0:
        if row[0] == df.min()['RMSE']:
            flag = 1
    else:
        flag = 1
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    return flag

#%%
def get_hyperparameters(x):
    """Get hyperparameters for solution `x`."""
    units = 2**int(x[0]*6 + 1)
    num_layers = int(x[1]*3)+1
    seq = int(x[2]*47 + 1)
    # lr = x[3]*2e-2 + 0.5e-3
    dense_units = 2**int(x[0]*7 + 1)
    params =  {
        'units': units,
        'num_layers': num_layers,
        'seq':seq,
        # 'lr':lr,
        'dense_units':dense_units
    }
    # print(params)
    return params


def get_data(num_feat,params):
    X_train,y_train,X_val,y_val,X_test_list ,y_test_list,scaler,Mids_test = get_dataset_alibaba_lstm_no_cluster(params['seq'],num_feat)
    X_test =list_to_array(X_test_list,params['seq'],num_feat)
    y_test = list_to_array(y_test_list,0,num_feat)
    return X_train,y_train,X_val,y_val,X_test ,y_test,scaler,X_test_list,y_test_list,Mids_test


class LSTMHyperparameterOptimization(Problem):
    def __init__(self, num_feat,num_epoc,batch_size,save_path):
        super().__init__(dimension=4, lower=0, upper=1)
        self.num_feat = num_feat
        self.save_path = save_path
        self.num_epoc = num_epoc
        self.batch_size = batch_size

    def _evaluate(self, x):
        if isinstance(x, dict):
            params = x
        else:
            params = get_hyperparameters(x)
        X_train,y_train,X_val,y_val,X_test ,y_test,scaler,X_test_list,y_test_list,Mids_test= get_data(self.num_feat,params)
        X_BB,Y_BB = get_BB_data(params['seq'],scaler)
        X_google,Y_google = get_google_data(params['seq'],scaler)

        output_dim = 1
        input_dim=(X_train.shape[1],X_train.shape[2])
        y_train = expand_dims(expand_dims(y_train))
        y_val = expand_dims(expand_dims(y_val))
        y_test = expand_dims(expand_dims(y_test))*scaler

        # model = get_EN_DE_LSTM_model(input_dim,output_dim,**params) #get_lstm_model
        model = get_en_de_lstm_model_attention(input_dim,output_dim,**params)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)

        callbacks_list = [EarlyStopping(monitor='val_loss', 
                                        patience=30, restore_best_weights=True),lr_scheduler]
        model.fit(X_train, y_train, epochs=self.num_epoc , 
                  batch_size=self.batch_size, verbose=0, shuffle=True, 
                  validation_data=(X_test,y_test),callbacks=callbacks_list)
        y_test_pred = (model.predict(X_test, verbose=0))*scaler
        row_alibaba = [RMSE(y_test,y_test_pred),MAE(y_test,y_test_pred)
                       ,MAPE(y_test,y_test_pred)]+[params['seq'],params['num_layers'],params['units']]
        flag_ali = log_results_LSTM(row_alibaba,self.save_path,'CEDL_Alibaba.csv')
        if flag_ali == 1:
            model.save(os.path.join(self.save_path,'Alibaba.keras'))
        
        y_pred_BB = (model.predict(X_BB, verbose=0))*scaler
        row_BB = [RMSE(Y_BB,y_pred_BB),MAE(Y_BB,y_pred_BB),
          MAPE(Y_BB,y_pred_BB)]+[params['seq'],params['num_layers'],params['units']]
        flag_BB = log_results_LSTM(row_BB,self.save_path,'CEDL_BB.csv')
        if flag_BB == 1:
            model.save(os.path.join(self.save_path,'BB.keras'))
        
        y_pred_google = (model.predict(X_google, verbose=0))*scaler
        row_google = [RMSE(Y_google,y_pred_google),MAE(Y_google,y_pred_google)
          ,MAPE(Y_google,y_pred_google)]+[params['seq'],params['num_layers'],params['units']]
        flag_google = log_results_LSTM(row_google,self.save_path,'CEDL_google.csv')
        if flag_google == 1:
            model.save(os.path.join(self.save_path,'Google.keras'))
        print("RMSE:",row_alibaba[0])
        return  row_alibaba[0]

#%%
from args import get_paths
base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()
# reconstructed_model = keras.models.load_model("my_model.keras")
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
num_feat = 2
run_search= 1
pop_size= 5
num_epoc = 2500
FF_itr = 15
batch_size = 2**9
alg_range = [2]

vb = 1
#%%
algorithm = CuckooSearch(population_size = pop_size) 

alg_name = algorithm.Name[0]+'_population_'+str(pop_size)+'_itr_'+str(FF_itr)
save_path = os.path.join(sav_path,alg_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
#%%
if run_search: 
    problem = LSTMHyperparameterOptimization(num_feat,num_epoc,batch_size,save_path)
    task = Task(problem, max_iters=FF_itr, optimization_type=OptimizationType.MINIMIZATION)

    
    best_params, best_mse = algorithm.run(task)
    
    best_para_save = get_hyperparameters(best_params)
    

    a_itr,b_itr = task.convergence_data()
    a_eval,b_eval = task.convergence_data(x_axis='evals')
    sav_dict_par = {'a_itr':a_itr,'b_itr':b_itr,'a_eval':a_eval,'b_eval':b_eval,'best_para_save':best_para_save}
    save_object(sav_dict_par,os.path.join(save_path,'Best_param'+alg_name+'.obj'))
    print('Best parameters:', best_para_save)
    task.plot_convergence(x_axis='evals')
    
    # plt.savefig(os.path.join(save_path,'Conv_FF_eval'+str(datatype_opt)+alg_name+'.png'))
    # plt.close()
    
    task.plot_convergence()
    
    plt.savefig(os.path.join(save_path,'Conv_FF_itr_n_feat'+str(num_feat)+alg_name+'.png'))
    plt.close()

