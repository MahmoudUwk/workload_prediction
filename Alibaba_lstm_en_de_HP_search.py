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
from Alibaba_helper_functions import loadDatasetObj,save_object,MAPE,MAE,RMSE,expand_dims
from Alibaba_fet_features_LSTM import get_dataset_alibaba_lstm
from niapy.algorithms.basic import CuckooSearch,MonkeyKingEvolutionV3
import warnings
warnings.filterwarnings('ignore')
def log_results_LSTM(row,save_path):

    save_name = 'results_LSTM_HP.csv'
    cols = ["RMSE", "MAE", "MAPE(%)","seq","num_layers","units","best epoch","n_cluster","num_feat"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    # print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    


def get_hyperparameters(x):
    """Get hyperparameters for solution `x`."""
    units = int(x[0]*116 + 10)
    num_layers = int(x[1]*6)+1
    seq = int(x[2]*15 + 1)
    lr = x[3]*2e-2 + 0.5e-3
    dense_units = int(x[0]*540 + 12)
    params =  {
        'units': units,
        'num_layers': num_layers,
        'seq':seq,
        'lr':lr,
        'dense_units':dense_units
    }
    # print(params)
    return params


def get_LSTM_model(input_dim,output_dim,units,num_layers,seq,lr,dense_units,name='LSTM_HP'):
    model = Sequential(name=name)
    state_falg = False
    model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = False,stateful=state_falg))
    model.add(RepeatVector(input_dim[0]))
    for dummy in range(num_layers-1):    
        model.add(LSTM(units=units,return_sequences = True,stateful=state_falg))
    # model.add(keras.layers.Attention())
    model.add(TimeDistributed(Dense(units,activation='relu')))
    model.add(Flatten())
    model.add(Dense(dense_units))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def get_classifier(x,input_dim,output_dim):
    """Get classifier from solution `x`."""
    if isinstance(x, dict):
        params = x
    else:
        params = get_hyperparameters(x)
    return get_LSTM_model(input_dim,output_dim,**params)

def get_data(x,cluster_num,num_feat):
    if isinstance(x, dict):
        params = x
    else:
        params = get_hyperparameters(x)
    X_train,y_train,X_val,y_val,X_test ,y_test,scaler,clusters = get_dataset_alibaba_lstm(params['seq'],cluster_num,num_feat)

    return X_train,y_train,X_val,y_val,X_test ,y_test,scaler,clusters


class LSTMHyperparameterOptimization(Problem):
    def __init__(self, num_feat,cluster_num,num_epoc):
        super().__init__(dimension=5, lower=0, upper=1)
        self.num_feat = num_feat
        self.cluster_num = cluster_num
        self.num_epoc = num_epoc

    def _evaluate(self, x):
        X_train,y_train,X_val,y_val,X_test ,y_test,scaler,clusters = get_data(x,self.cluster_num,self.num_feat)
        output_dim = 1
        input_dim=(X_train.shape[1],X_train.shape[2])
        y_train = expand_dims(expand_dims(y_train))
        y_test = expand_dims(expand_dims(y_test))
        model = get_classifier(x,input_dim,output_dim)
        out_put_model = [layer.output_shape for c,layer in enumerate(model.layers) if c==len(model.layers)-1][0][1]
        # print(model.summary())
        assert(out_put_model==output_dim)
        # print(X_train.shape,y_train.shape)
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
        model.fit(X_train, y_train, epochs=self.num_epoc , batch_size=2**10, verbose=0, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
        return  model.evaluate(X_test,y_test)

#%%
# base_path = "data/"


num_feat = 1
cluster_nums = range(4) #['1s','1T','15T','30T','home','1s']
run_search= 0
pop_size= 10
num_epoc = 5
FF_itr = 15
alg_range = [0]#range(3)
sav_path_general = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/results/lstm_search_alg"
test_lens = np.zeros(4)
#%%
for alg_all in alg_range:
    for cluster_num in cluster_nums:
    
        save_path = sav_path_general+"/cluster_"+str(cluster_num)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if alg_all == 0:
            # 
            algorithm = Mod_FireflyAlgorithm.Mod_FireflyAlgorithm(population_size = pop_size)
        elif alg_all == 1:
            algorithm = FireflyAlgorithm(population_size = pop_size)
        else:
            algorithm = CuckooSearch(population_size = pop_size) 
            # algorithm = MonkeyKingEvolutionV3(population_size = pop_size) 
            
    #%%
        if run_search: 
            problem = LSTMHyperparameterOptimization(num_feat,cluster_num,num_epoc)
            task = Task(problem, max_iters=FF_itr, optimization_type=OptimizationType.MINIMIZATION)
    
            
            best_params, best_mse = algorithm.run(task)
            
            best_para_save = get_hyperparameters(best_params)
            

            a_itr,b_itr = task.convergence_data()
            a_eval,b_eval = task.convergence_data(x_axis='evals')
            sav_dict_par = {'a_itr':a_itr,'b_itr':b_itr,'a_eval':a_eval,'b_eval':b_eval,'best_para_save':best_para_save}
            save_object(sav_dict_par,os.path.join(save_path,'Best_param'+algorithm.Name[0]+'.obj'))
            print('Best parameters:', best_para_save)
            task.plot_convergence(x_axis='evals')
            
            # plt.savefig(os.path.join(save_path,'Conv_FF_eval'+str(datatype_opt)+algorithm.Name[0]+'.png'))
            # plt.close()
            
            task.plot_convergence()
            
            plt.savefig(os.path.join(save_path,'Conv_FF_itr_n_feat'+str(num_feat)+algorithm.Name[0]+'.png'))
            plt.close()
    
    
        #%%



        best_params = loadDatasetObj(os.path.join(save_path,'Best_param'+algorithm.Name[0]+'.obj'))['best_para_save']

        X_train,y_train,X_val,y_val,X_test ,y_test,scaler,clusters = get_data(best_params,cluster_num,num_feat)
        
        test_lens[cluster_num] = X_test.shape[0]
        
        y_train = expand_dims(expand_dims(y_train))
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        
        model = get_classifier(best_params,input_dim,output_dim)
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
        
        history = model.fit(X_train, y_train, epochs=num_epoc, batch_size=2**8, verbose=0, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
        

        # model.save(algorithm.Name[0]+'_n_feat_'+str(num_feat))
        best_epoch = np.argmin(history.history['val_loss'])
     
        y_test = y_test*scaler  
        y_test_pred = (model.predict(X_test))*scaler
        rmse = RMSE(y_test,y_test_pred)
        mae = MAE(y_test,y_test_pred)
        mape = MAPE(y_test,y_test_pred)
        print(rmse,mae,mape)
        alg_name = algorithm.Name[0]
        
    #%%
        row = [rmse,mae,mape,best_params['seq'],best_params['num_layers'],best_params['units'],best_epoch,cluster_num,num_feat]


        log_results_LSTM(row,sav_path_general)
        
        filename = os.path.join(save_path,alg_name+'.obj')
        obj = {'y_test':y_test,'y_test_pred':y_test_pred}
        save_object(obj, filename)
    
#%%

#%%
save_name = sav_path_general+'/results_LSTM_HP.csv'

df = pd.read_csv(save_name)
# print(df['RMSE'])
# print(df.sort_values(by=['RMSE'])[:5])
#3.32 base paper
RMSE_all = np.sum(df.groupby(by='n_cluster').min()['RMSE']*np.array(test_lens)/sum(test_lens))
print('RMSE_all: ',RMSE_all)
l_u=[]
for n,cluster_dat in df.groupby(by='n_cluster'):

    list_i = cluster_dat[['num_layers','units','seq']].iloc[cluster_dat['RMSE'].argmin()]
    l_u.append(list_i)
