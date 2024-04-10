
# from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from keras.layers import  LSTM,Dense#,Bidirectional
from keras.layers import TimeDistributed,Flatten
from keras.layers import RepeatVector
from keras.models import  Sequential #,load_model
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
import os
# import keras
from keras.callbacks import EarlyStopping
# from Alibaba_helper_functions import get_Mid

from Alibaba_helper_functions import loadDatasetObj,save_object,MAPE,MAE,RMSE,expand_dims,list_to_array,get_EN_DE_LSTM_model
from Alibaba_fet_features_LSTM_no_cluster import get_dataset_alibaba_lstm_no_cluster

base_path = "data/"

sav_path = base_path+"results/lstm_en_de"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

#%%


def log_results_LSTM(row,save_path):

    save_name = 'results_LSTM_en_de.csv'
    cols = ["RMSE", "MAE", "MAPE(%)","seq","num_layers","units","best epoch","train_time(min)","num_feat"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    # print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    

#%%
alg_name = 'LSTM'
lr = 0.001
opt_chosen=Adam(learning_rate=lr)


callback_falg = 1

output_dim = 1
batch_size_n = 2**9

num_units = [50]#[35]#[8,10,15]
num_layers_all = [2,3]

losses = ['mse']#,'mae']
seq_length_all = [6,9]#12
val_split_size = 0.3
num_features = [1]#range(1,7,1)
dense_units = 256
epochs_num = 2000
np.random.seed(7)
for num_feat in num_features:
    for seq_length in seq_length_all:
        X_train,y_train,X_val,y_val,X_test_list ,y_test_list,scaler,Mids_test = get_dataset_alibaba_lstm_no_cluster(seq_length,num_feat)
        X_test =list_to_array(X_test_list,seq_length,num_feat)
        y_test = list_to_array(y_test_list,0,num_feat)

    
        ind_rand = np.random.permutation(len(X_train))
        X_train = X_train[ind_rand]
        y_train = y_train[ind_rand]
        y_test = y_test*scaler
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
    
    
        for num_layers in num_layers_all:   
            for units in num_units:
                model = get_EN_DE_LSTM_model(input_dim,output_dim,units,num_layers,dense_units)         
                model.compile(optimizer=opt_chosen, loss='mse')
                print(model.summary())
                print('start training')
                start_train = time.time()
                if callback_falg:
                    callbacks_list = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
                    if X_val.size ==0:
                        history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_split=val_split_size,callbacks=callbacks_list)
                    else:
                        history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
                else:
                    print('Stop')
                #%%
                end_train = time.time()
                print('End training')
                train_time = (end_train - start_train)/60
                if callback_falg:
                    # model.set_weights(checkpoint.best_weights)
                    best_epoch =np.argmin(history.history['val_loss'])
                else:
                    best_epoch = 0
                # model.save(filepath)
                #%%
                
                start_test = time.time()
                y_test_pred = model.predict(X_test)*scaler
                end_test = time.time()
                test_time = end_test - start_test
                #%%
                rmse = RMSE(y_test,y_test_pred)
                mae = MAE(y_test,y_test_pred)
                mape = MAPE(y_test,y_test_pred)
                print(rmse,mae,mape)
                # best_epoch = epochs_num

                row = [rmse,mae,mape,seq_length,num_layers,units,best_epoch,train_time,num_feat]
         
                log_results_LSTM(row,sav_path)


#%%
save_name = 'results_LSTM_en_de.csv'

df = pd.read_csv(os.path.join(sav_path,save_name))
print(df['RMSE'])










