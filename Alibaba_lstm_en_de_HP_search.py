import matplotlib.pyplot as plt
import os
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, cross_val_score
from keras.layers import Dense,LSTM,RepeatVector,TimeDistributed,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM

from niapy.problems import Problem
from niapy.task import Task, OptimizationType
import numpy as np
import pandas as pd
from niapy.algorithms.modified import Mod_FireflyAlgorithm
from niapy.algorithms.basic import FireflyAlgorithm
from Alibaba_helper_functions import loadDatasetObj,save_object,MAPE,MAE,RMSE,expand_dims,list_to_array,get_EN_DE_LSTM_model
from Alibaba_fet_features_LSTM_no_cluster import get_dataset_alibaba_lstm_no_cluster
from niapy.algorithms.basic import CuckooSearch,MonkeyKingEvolutionV3
import warnings
import time
warnings.filterwarnings('ignore')
np.random.seed(7)
def log_results_LSTM(row,save_path):

    save_name = 'results_LSTM_en_de_HP_seach.csv'
    cols = ["RMSE", "MAE", "MAPE(%)","seq","num_layers","units","best epoch","num_feat",'algorithm_name','batch_size','test_num','alg']

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    flag = 0
    if len(df[df['alg']==row[-1]])!=0:
        if row[0] == df[df['alg']==row[-1]].min()['RMSE']:
            flag = 1
    else:
        flag = 1
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    return flag
    # print(df)
#%%


def get_hyperparameters(x):
    """Get hyperparameters for solution `x`."""
    units = int(x[0]*116 + 10)
    num_layers = int(x[1]*6)+1
    seq = int(x[2]*23 + 1)
    lr = x[3]*2e-2 + 0.5e-3
    dense_units = int(x[0]*600 + 12)
    params =  {
        'units': units,
        'num_layers': num_layers,
        'seq':seq,
        'lr':lr,
        'dense_units':dense_units
    }
    # print(params)
    return params




def get_classifier(x,input_dim,output_dim):
    """Get classifier from solution `x`."""
    if isinstance(x, dict):
        params = x
    else:
        params = get_hyperparameters(x)
    return get_EN_DE_LSTM_model(input_dim,output_dim,**params)

def get_data(x,num_feat):
    if isinstance(x, dict):
        params = x
    else:
        params = get_hyperparameters(x)
    X_train,y_train,X_val,y_val,X_test_list ,y_test_list,scaler,Mids_test = get_dataset_alibaba_lstm_no_cluster(params['seq'],num_feat)
    X_test =list_to_array(X_test_list,params['seq'],num_feat)
    y_test = list_to_array(y_test_list,0,num_feat)
    return X_train,y_train,X_val,y_val,X_test ,y_test,scaler,X_test_list,y_test_list,Mids_test


class LSTMHyperparameterOptimization(Problem):
    def __init__(self, num_feat,num_epoc):
        super().__init__(dimension=5, lower=0, upper=1)
        self.num_feat = num_feat

        self.num_epoc = num_epoc

    def _evaluate(self, x):
        X_train,y_train,X_val,y_val,X_test ,y_test,scaler,_,_,_ = get_data(x,self.num_feat)
        ind_rand = np.random.permutation(len(X_train))
        X_train = X_train[ind_rand]
        y_train = y_train[ind_rand]
        output_dim = 1
        input_dim=(X_train.shape[1],X_train.shape[2])
        y_train = expand_dims(expand_dims(y_train))
        y_val = expand_dims(expand_dims(y_val))
        y_test = expand_dims(expand_dims(y_test))
        # print(y_train.shape,y_val.shape,y_test.shape)
        model = get_classifier(x,input_dim,output_dim)
        out_put_model = [layer.output_shape for c,layer in enumerate(model.layers) if c==len(model.layers)-1][0][1]
        # print(model.summary())
        assert(out_put_model==output_dim)
        # print(X_train.shape,y_train.shape)
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
        model.fit(X_train, y_train, epochs=self.num_epoc , batch_size=2**12, verbose=0, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
        return  model.evaluate(X_test,y_test)

#%%
from args import get_paths
base_path,processed_path,_,_,feat_stats_step3,sav_path_general = get_paths()
if not os.path.exists(sav_path_general):
    os.makedirs(sav_path_general)
num_feat = 2
num_feat_list = [2]
batch_list = range(9,10)
run_search= 0
pop_size= 10
num_epoc = 2500
FF_itr = 15
alg_range = [2]

vb = 1
#%%
algorithm = CuckooSearch(population_size = pop_size) 

alg_name = algorithm.Name[0]#+'_population_'+str(pop_size)+'_itr_'+str(FF_itr)
save_path = os.path.join(sav_path_general,alg_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
#%%
if run_search: 
    problem = LSTMHyperparameterOptimization(num_feat,num_epoc)
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


#%%
for batch_pow in batch_list:
    for num_feat in num_feat_list: 
        batch_size = 2**batch_pow
        best_params = loadDatasetObj(os.path.join(save_path,'Best_param'+alg_name+'.obj'))['best_para_save']
        
        X_train,y_train,X_val,y_val,X_test ,y_test,scaler,X_test_list,y_test_list,Mids_test = get_data(best_params,num_feat)
        print(X_train.shape,X_val.shape,X_test.shape)


        
        y_train = expand_dims(expand_dims(y_train))
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        
        model = get_classifier(best_params,input_dim,output_dim)
        
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
        
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
        
        start_train = time.time()
        history = model.fit(X_train, y_train, epochs=num_epoc, batch_size=batch_size, verbose=vb, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
        end_train = time.time()
        train_time = (end_train - start_train)/60

        # model.save(alg_name+'_n_feat_'+str(num_feat))
        best_epoch = np.argmin(history.history['val_loss'])
     
        y_test = y_test*scaler  
        
        y_test_pred = (model.predict(X_test))*scaler

        rmse = RMSE(y_test,y_test_pred)
        mae = MAE(y_test,y_test_pred)
        mape = MAPE(y_test,y_test_pred)
        # print(rmse,mae,mape)
        
        
        
    #%%
        row = [rmse,mae,mape,best_params['seq'],best_params['num_layers'],best_params['units'],best_epoch,num_feat,alg_name,batch_size,X_test.shape[0],alg_name]
        name_sav = ""
        for n in row:
            name_sav = name_sav+str(n)+"_" 

        flag = log_results_LSTM(row,sav_path_general)
        save_path_dat = base_path+'/pred_results_all'
        if not os.path.exists(save_path_dat):
            os.makedirs(save_path_dat)
        filename = os.path.join(save_path_dat,alg_name+'.obj')
        if flag == 1:
            y_test_pred_list = []
            rmse_list = []
            start_test = time.time()
            for c,test_sample in enumerate(X_test_list):
                pred_i = (model.predict(test_sample))
                y_test_pred_list.append(pred_i*scaler)
                rmse_i_list = RMSE(y_test_list[c]*scaler,pred_i*scaler)
                y_test_list[c] = y_test_list[c]*scaler
                rmse_list.append(rmse_i_list)
            end_test = time.time()
            test_time = end_test - start_test
            val_loss = history.history['val_loss']
            train_loss = history.history['loss']
            obj = {'test_time':test_time,'train_time':train_time,'y_test':y_test_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list),'best_params':best_params,'Mids_test':Mids_test,'val_loss':val_loss,'train_loss':train_loss}
            save_object(obj, filename)



#%%
# save_name = sav_path_general+'/results_LSTM_en_de_HP_seach_no_cluster5.csv'

# df = pd.read_csv(save_name)
# print(df)

# df.groupby('alg').min()['RMSE']

# ind1 = df[df['alg']=='FireflyAlgorithm']['RMSE'].argmin()
# df[df['alg']=='FireflyAlgorithm'].iloc[ind1,:]
# # 3.32 base paper

