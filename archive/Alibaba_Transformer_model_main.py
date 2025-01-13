from Alibaba_Transformer_model import build_model
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from keras.layers import  LSTM,Dense#,Bidirectional
from keras.models import  Sequential #,load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
import os
import keras
# from keras.callbacks import EarlyStopping
import pickle
from Alibaba_fet_features_LSTM_no_cluster import get_dataset_alibaba_lstm_no_cluster

from Alibaba_helper_functions import loadDatasetObj,save_object,MAPE,MAE,RMSE,expand_dims,list_to_array,get_EN_DE_LSTM_model

#%%
def log_results_LSTM(row,save_path):

    save_name = 'results_transformer_all_data.csv'
    cols = [ "RMSE", "MAE", "MAPE(%)","seq","num_transformer_blocks","mlp_units","num_heads","head_size","best epoch","train_time(min)","alg_name"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df['RMSE'])
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    
#%%
base_path = "data/"

sav_path_general = base_path+"LSTM_results/lstm_en_de_search_alg"
if not os.path.exists(sav_path_general):
    os.makedirs(sav_path_general)
    
sav_path = base_path+"results/lstm"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

alg_name = 'Transformer'
lr = 0.01
opt_chosen=Adam(learning_rate=lr)

epochs_num = 500
drop_out = 0
callback_falg = 1

output_dim = 1
batch_size_n = 2**11

mlp_units_all = [[64]]#[35]#[8,10,15]
num_transformer_blocks_all = [2]
head_size_all = [64]
num_heads_all = [4]
losses = ['mse']#,'mae']

seq_length_all = [12]


num_feat = 3
for seq_length in seq_length_all:
    X_train,y_train,X_val,y_val,X_test_list ,y_test_list,scaler,Mids_test = get_dataset_alibaba_lstm_no_cluster(seq_length,num_feat)
    X_test =list_to_array(X_test_list,seq_length,num_feat)
    y_test = list_to_array(y_test_list,0,num_feat)    
    ind_rand = np.random.permutation(len(X_train))
    X_train = X_train[ind_rand]
    y_train = y_train[ind_rand]

    input_dim=(X_train.shape[1],X_train.shape[2])
    print(X_train.shape)
    print(input_dim,output_dim)
    y_train = expand_dims(expand_dims(y_train))
    y_val = expand_dims(expand_dims(y_val))
    if len(X_train.shape)<3:
        X_train = expand_dims(X_train)
        X_val = expand_dims(X_val)
        X_test = expand_dims(X_test)
    print(X_train.shape,y_train.shape,X_test.shape)
 
    
    for loss in losses:
        for num_transformer_blocks in num_transformer_blocks_all:   
            for mlp_units in mlp_units_all:
                for num_heads in num_heads_all:
                    for head_size in head_size_all:
                        
                        model = build_model(
                            input_dim,
                            head_size=head_size,
                            num_heads=num_heads,
                            ff_dim=3,
                            num_transformer_blocks=num_transformer_blocks,
                            mlp_units=mlp_units,
                            mlp_dropout=0,
                            dropout=0,
                        )
                        
                        model.compile(optimizer=opt_chosen, loss=loss)
                        print(model.summary())
                        print('start training')
                        start_train = time.time()
                
                        callbacks_list = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
                        history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)

                        end_train = time.time()
                        print('End training')
                        train_time = (end_train - start_train)/60

                        best_epoch =np.argmin(history.history['val_loss'])

    
                        
                        start_test = time.time()
                        y_test_pred = model.predict(X_test)*scaler
                        end_test = time.time()
                        test_time = end_test - start_test
                        #%%
                        y_test = y_test*scaler  
                        y_test_pred = (model.predict(X_test))*scaler
                        rmse = RMSE(y_test,y_test_pred)
                        mae = MAE(y_test,y_test_pred)
                        mape = MAPE(y_test,y_test_pred)
                        row = [rmse,mae,mape,seq_length,num_transformer_blocks,mlp_units,num_heads,head_size,best_epoch,train_time,alg_name]

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
                            for c,test_sample in enumerate(X_test_list):
                                pred_i = (model.predict(test_sample))
                                y_test_pred_list.append(pred_i*scaler)
                                rmse_i_list = RMSE(y_test_list[c]*scaler,pred_i*scaler)
                                y_test_list[c] = y_test_list[c]*scaler
                                rmse_list.append(rmse_i_list)
                                
                            val_loss = history.history['val_loss']
                            train_loss = history.history['loss']
                            obj = {'y_test':y_test_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list),'Mids_test':Mids_test,'val_loss':val_loss,'train_loss':train_loss}
                            save_object(obj, filename)


#%%
save_name = sav_path_general+'/results_transformer_all_data.csv'

df = pd.read_csv(save_name)
print(df)









