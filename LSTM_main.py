import os
from keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from utils_WLP import save_object, expand_dims,get_en_de_lstm_model_attention,RMSE,MAPE,MAE
from utils_WLP import switch_model_CSA,get_dict_option,get_lstm_model,model_serv,get_best_lsmt_para,get_en_de_lstm_model_attentionV2
import time
flag_datasets = [0]
flag_models = [2]
data_set_flags = ['Alibaba','BB']
seqs = [29,20,32]
models_lstm = ['EnDeAtt','LSTM','TST_LSTM']
scaler = 100
num_epoc = 700

batch_sizes = [2**6,2**7]

# lr = 0.001

for flag_dataset in flag_datasets:
    for flag_model in flag_models:
        
        seq = seqs[flag_model]
        best_params =    get_best_lsmt_para(flag_model,flag_dataset)
        if data_set_flags[flag_dataset] == 'Alibaba':
            from args import get_paths
        elif data_set_flags[flag_dataset]=='BB':
            from args_BB import get_paths
        
        base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        

        
        train_dict , val_dict, test_dict = get_dict_option(data_set_flags[flag_dataset],seq)
        
        
        X_train = train_dict['X']
        y_train = train_dict['Y']
        
        X_val = val_dict['X']
        y_val = val_dict['Y']
        
        X_test_list = test_dict['X_list']
        y_test_list = test_dict['Y_list']
        X_test = test_dict['X']
        y_test = test_dict['Y']
        Mids_test = test_dict['M_ids']
        
        output_dim = 1
        input_dim=(X_train.shape[1],X_train.shape[2])
        y_train = expand_dims(expand_dims(y_train))
        y_val = expand_dims(expand_dims(y_val))
        #%%
        batch_size = batch_sizes[flag_dataset]
        lr = 0.001*batch_size/2**7
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=2e-4)
    
        model  = switch_model_CSA(models_lstm[flag_model],input_dim,output_dim,best_params)
        model.compile(optimizer=Adam(learning_rate=lr), loss='MAE') 
     
        callbacks_list = [EarlyStopping(monitor='val_loss', 
                            patience=15, restore_best_weights=True),reduce_lr]
        
        start_train = time.time()

        history = model.fit(X_train, y_train, epochs=num_epoc , 
                  batch_size=batch_size, verbose=2, shuffle=True, 
                  validation_data=(X_val,y_val),callbacks=callbacks_list)
        
        end_train = time.time()
        train_time = (end_train - start_train)/60
        #%%
        y_test_pred = (model.predict(X_test))*scaler
        row_alibaba = [RMSE(y_test*scaler,y_test_pred),MAE(y_test*scaler,y_test_pred)
                       ,MAPE(y_test*scaler,y_test_pred)]
        
        
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
        print(row_alibaba)
