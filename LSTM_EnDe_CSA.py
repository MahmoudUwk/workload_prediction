import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping
import time
import numpy as np
from niapy.problems import Problem
from niapy.task import Task, OptimizationType
import pandas as pd
from utils_WLP import save_object , model_serv,loadDatasetObj,save_object,MAPE,MAE,RMSE,get_lstm_model
from utils_WLP import get_data_CSA,switch_model_CSA , switch_para_CSA, expand_dims,list_to_array,get_en_de_lstm_model_attention,get_dict_option,log_results_EB0
from niapy.algorithms.basic import CuckooSearch


#%%

class LSTMHyperparameterOptimization(Problem):
    def __init__(self,data_set, num_epoc,batch_size,save_path,model_name,dim,func_para):
        self.dim = dim
        super().__init__(dimension=self.dim , lower=0, upper=1)
        self.func_para = func_para
        self.save_path = save_path
        self.num_epoc = num_epoc
        self.batch_size = batch_size
        self.model_name = model_name
        self.data_set = data_set

    def _evaluate(self, x):
        if isinstance(x, dict):
            params = x
        else:
            params = func_para(x)
        X_train,y_train,X_val,y_val,X_test ,y_test,scaler,X_test_list,y_test_list,Mids_test= get_data_CSA(params,self.data_set)

        output_dim = 1
        input_dim=(X_train.shape[1],X_train.shape[2])

        model = switch_model_CSA(self.model_name,input_dim,output_dim,params)

        callbacks_list = [EarlyStopping(monitor='val_loss', 
                                        patience=15, restore_best_weights=True)]#,lr_scheduler]
        start_train = time.time()
        history = model.fit(X_train, y_train, epochs=self.num_epoc , 
                  batch_size=self.batch_size, verbose=2, shuffle=True, 
                  validation_data=(X_val,y_val),callbacks=callbacks_list)
        end_train = time.time()
        train_time = (end_train - start_train)/60
        y_test_pred = (model.predict(X_test))*scaler
        row_log= [RMSE(y_test*scaler,y_test_pred),MAE(y_test*scaler,y_test_pred)
                       ,MAPE(y_test*scaler,y_test_pred)]+list((params).values())
        
        cols = ['RMSE','MAE','MAPE(%)'] + list((params).keys())
        save_name = self.data_set+self.model_name
        flag_sav = log_results_EB0(row_log,cols,os.path.join(self.save_path,save_name+'.csv'))
        if flag_sav == 1:
            filename = os.path.join(sav_path,save_name+'.obj')
            start_test = time.time()
            y_test_pred_list,rmse_list = model_serv(X_test_list,y_test_list,model,scaler,batch_size)
            end_test = time.time()
            test_time = end_test - start_test
            val_loss = history.history['val_loss']
            train_loss = history.history['loss']
            obj = {'test_time':test_time,'train_time':train_time
                   ,'y_test':y_test_list,'y_test_pred':y_test_pred_list
                   ,'scaler':scaler,'rmse_list':np.array(rmse_list)
                   ,'best_params':params,'Mids_test':Mids_test,'val_loss':val_loss
                   ,'train_loss':train_loss}
            save_object(obj, filename)
    
        print("RMSE:",row_log[0])
        return  row_log[0]
#%%
flag_dataset = 2
data_set = ['Alibaba','google','BB'][flag_dataset]
flag_model = 2
model_name =  ['EnDeAtt','LSTM','TST_LSTM'][flag_model]

func_para = switch_para_CSA(model_name)
dim = len(func_para([]))




if data_set== 'Alibaba':
    from args import get_paths
elif data_set=='google':
    from args_google import get_paths
elif data_set=='BB':
    from args_BB import get_paths

base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()

sav_path = os.path.join(sav_path,model_name)
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

run_search= 1
pop_size= 5
num_epoc = 700
FF_itr = 15
batch_size = 2**12

vb = 1
#%%
algorithm = CuckooSearch(population_size = pop_size) 

alg_name = algorithm.Name[0]+'_val_population_'+str(pop_size)+'_itr_'+str(FF_itr)
save_path = os.path.join(sav_path,alg_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
#%%
if run_search: 
    problem = LSTMHyperparameterOptimization(data_set,num_epoc
                                 ,batch_size,save_path,model_name,dim,func_para)
    task = Task(problem, max_iters=FF_itr, optimization_type=OptimizationType.MINIMIZATION)

    
    best_params, best_mse = algorithm.run(task)
    
    best_para_save = func_para(best_params)
    

    a_itr,b_itr = task.convergence_data()
    a_eval,b_eval = task.convergence_data(x_axis='evals')
    sav_dict_par = {'a_itr':a_itr,'b_itr':b_itr,'a_eval':a_eval,'b_eval':b_eval,'best_para_save':best_para_save}
    save_object(sav_dict_par,os.path.join(save_path,'Best_param'+alg_name+'.obj'))
    print('Best parameters:', best_para_save)
    task.plot_convergence(x_axis='evals')
    
    # plt.savefig(os.path.join(save_path,'Conv_FF_eval'+str(datatype_opt)+alg_name+'.png'))
    # plt.close()
    
    task.plot_convergence()
    
    plt.savefig(os.path.join(save_path,'Conv_FF_itr_n_feat'+data_set+model_name+alg_name+'.png'))
    plt.close()

