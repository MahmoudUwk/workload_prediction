
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
from Alibaba_fet_features_LSTM import get_dataset_alibaba_lstm
from Alibaba_helper_functions import loadDatasetObj,save_object,MAPE,MAE,RMSE,expand_dims


base_path = "data/"

sav_path = base_path+"results/lstm"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

#%%
def get_LSTM_model(input_dim,output_dim,units,num_layers,name='LSTM_HP'):
    model = Sequential(name=name)
    state_falg = False
    model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = False,stateful=state_falg))
    model.add(RepeatVector(input_dim[0]))
    for dummy in range(num_layers-1):    
        model.add(LSTM(units=units,return_sequences = True,stateful=state_falg))
    # model.add(keras.layers.Attention())
    model.add(TimeDistributed(Dense(units,activation='relu')))
    model.add(Flatten())
    model.add(Dense(output_dim))
    return model

def log_results_LSTM(row,save_path):

    save_name = 'results_LSTM_en6.csv'
    cols = ["loss", "RMSE", "MAE", "MAPE(%)","seq","num_layers","units","best epoch","train_time(min)","n_cluster","num_feat"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    # print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    

#%%
alg_name = 'LSTM'
lr = 0.0084
opt_chosen=Adam(learning_rate=lr)

drop_out = 0
callback_falg = 1

output_dim = 1
batch_size_n = 2**8

num_units = [10]#[35]#[8,10,15]
num_layers_all = [2]

losses = ['mse']#,'mae']
seq_length_all = [9]#12
val_split_size = 0.3
num_features = [1]#range(1,7,1)
cluster_nums = [1]#range(4)
epochs_num = 2000
for cluster_num  in cluster_nums:
    for num_feat in num_features:
        for seq_length in seq_length_all:
            X_train,y_train,X_val,y_val,X_test ,y_test,scaler,clusters = get_dataset_alibaba_lstm(seq_length,cluster_num,num_feat)
        
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
                    
                    model = get_LSTM_model(input_dim,output_dim,units,num_layers)         
                    model.compile(optimizer=opt_chosen, loss='mse')
                    print(model.summary())
                    print('start training')
                    start_train = time.time()
                    if callback_falg:
                        callbacks_list = [EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)]
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
                    clus_des = str(cluster_num)+" out of " + str(len(clusters))
                    row = ['mse',rmse,mae,mape,seq_length,num_layers,units,best_epoch,train_time,clus_des,num_feat]
             
                    log_results_LSTM(row,sav_path)


#%%
save_name = 'results_LSTM_en6.csv'

df = pd.read_csv(os.path.join(sav_path,save_name))
print(df['RMSE'])
# print(df.sort_values(by=['RMSE'])[:5])

RMSE_all = np.sum(df.groupby(by='n_cluster').min()['RMSE']*np.array(clusters)/sum(clusters))
l_u = []
for n,cluster_dat in df.groupby(by='n_cluster'):

    (a,b) = cluster_dat[['num_layers','units']].iloc[cluster_dat['RMSE'].argmin()]
    l_u.append((a,b))










