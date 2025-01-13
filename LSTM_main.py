import os
from keras.callbacks import EarlyStopping
import numpy as np
from utils_WLP import save_object, expand_dims,get_en_de_lstm_model_attention
from utils_WLP import get_dict_option,get_lstm_model,model_serv
import time
flag_dataset = 1
flag_model = 0
flag_datasets = [1,2]
flag_models = [0,1]
data_set_flags = ['Alibaba','google','BB']

models_lstm =  ['EnDeAtt','LSTM']
for flag_dataset in flag_datasets:
    for flag_model in flag_models:

        
        
        if data_set_flags[flag_dataset] == 'Alibaba':
            from args import get_paths
        elif data_set_flags[flag_dataset]=='google':
            from args_google import get_paths
        elif data_set_flags[flag_dataset]=='BB':
            from args_BB import get_paths
        
        base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        
        best_params =  {'units': 256,
         'num_layers': 2,
         'seq': 20,
         'dense_units': 256}
        
        train_dict , val_dict, test_dict = get_dict_option(data_set_flags[flag_dataset],best_params['seq'])
        
        
        X_train = train_dict['X']
        y_train = train_dict['Y']
        
        X_val = val_dict['X']
        y_val = val_dict['Y']
        
        X_test_list = test_dict['X_list']
        y_test_list = test_dict['Y_list']
        Mids_test = test_dict['M_ids']
        
        output_dim = 1
        input_dim=(X_train.shape[1],X_train.shape[2])
        y_train = expand_dims(expand_dims(y_train))
        y_val = expand_dims(expand_dims(y_val))
        #%%
        scaler = 100
        num_epoc = 700
        batch_size = 2**6
        if flag_model==0:
            model= get_en_de_lstm_model_attention(input_dim,output_dim,**best_params)
        elif flag_model==1:
            model = get_lstm_model(input_dim,output_dim,**best_params)
        
        callbacks_list = [EarlyStopping(monitor='val_loss', 
                            patience=15, restore_best_weights=True)]
        
        start_train = time.time()
        history = model.fit(X_train, y_train, epochs=num_epoc , 
                  batch_size=batch_size, verbose=2, shuffle=True, 
                  validation_data=(X_val,y_val),callbacks=callbacks_list)
        
        end_train = time.time()
        train_time = (end_train - start_train)/60
        #%%
        save_name = data_set_flags[flag_dataset]+models_lstm[flag_model]
        filename = os.path.join(sav_path,save_name+'.obj')
        y_test_pred_list = []
        rmse_list = []
        start_test = time.time()
        
        y_test_pred_list,rmse_list = model_serv(X_test_list,y_test_list,model,scaler,batch_size)
        
        
        end_test = time.time()
        test_time = end_test - start_test
        val_loss = history.history['val_loss']
        train_loss = history.history['loss']
        obj = {'test_time':test_time,'train_time':train_time
               ,'y_test':y_test_list,'y_test_pred':y_test_pred_list
               ,'scaler':scaler,'rmse_list':np.array(rmse_list)
               ,'best_params':best_params,'Mids_test':Mids_test,'val_loss':val_loss
               ,'train_loss':train_loss}
        save_object(obj, filename)
