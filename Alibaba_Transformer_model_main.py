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
# from Alibaba_helper_functions import get_Mid
from Alibaba_fet_features_LSTM import get_dataset_alibaba_lstm

#%%
def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))

def RMSE(test,pred):
    return np.sqrt(np.mean((np.squeeze(test) - np.squeeze(pred))**2))

def MAE(test,pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test)))

def MAPE(test,pred):
    ind = np.where(test!=0)[0].flatten()
    return 100*np.mean(np.abs(np.squeeze(pred[ind]) - np.squeeze(test[ind]))/np.abs(np.squeeze(test[ind])))

def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

def log_results_LSTM(row,save_path):

    save_name = 'results_transformer_all_data.csv'
    cols = ["loss", "RMSE", "MAE", "MAPE(%)","seq","num_transformer_blocks","mlp_units","num_heads","head_size","best epoch","train_time(min)","n_cluster"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df['RMSE'])
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    
#%%
base_path = "data/"

sav_path = base_path+"results/lstm"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

alg_name = 'Transformer'
lr = 0.0001
opt_chosen=Adam(learning_rate=lr)

epochs_num = 500
drop_out = 0
callback_falg = 1

output_dim = 1
batch_size_n = 2**9

mlp_units_all = [[64]]#[35]#[8,10,15]
num_transformer_blocks_all = [2]
head_size_all = [64]
num_heads_all = [4]
losses = ['mse']#,'mae']
seq_length_all = [3]
cluster_nums = range(4) #['all'] #
val_split_size = 0.3
for seq_length in seq_length_all:
    for cluster_num  in cluster_nums:
    
        X_train,y_train,X_val,y_val,X_test ,y_test,scaler,clusters = get_dataset_alibaba_lstm(seq_length,cluster_num)
    
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
                            if callback_falg:
                                callbacks_list = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
                                if X_val.size ==0:
                                    history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_split=val_split_size,callbacks=callbacks_list)
                                else:
                                    history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
                            else:
                                print('Stop')
                                # history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True)#, validation_split=0.2,callbacks=callbacks_list)
        
                            end_train = time.time()
                            print('End training')
                            train_time = (end_train - start_train)/60
                            if callback_falg:
                                best_epoch =np.argmin(history.history['val_loss'])
                            else:
                                best_epoch = 0
        
                            
                            start_test = time.time()
                            y_test_pred = model.predict(X_test)*scaler
                            end_test = time.time()
                            test_time = end_test - start_test
                            #%%
                            rmse = RMSE(y_test,y_test_pred)
                            mae = MAE(y_test,y_test_pred)
                            mape = MAPE(y_test,y_test_pred)
                            
                            # best_epoch = epochs_num
                            if cluster_num =='all':
                                clus_des='all'
                            else:
                                clus_des = str(cluster_num)+" out of " + str(len(clusters))
                            row = [loss,rmse,mae,mape,seq_length,num_transformer_blocks,mlp_units,num_heads,head_size,best_epoch,train_time,clus_des]
                     
                            log_results_LSTM(row,sav_path)
                            print(rmse,mae,mape)

#%%
save_name = 'results_transformer_all_data.csv'
clusters
df = pd.read_csv(os.path.join(sav_path,save_name))
# print(df['RMSE'].min())
print(df.sort_values(by=['RMSE'])[:5])
# RMSE_all = df.groupby(by='n_cluster').min()['RMSE'].mean() 
# print(RMSE_all)
# l_u = []
# for n,cluster_dat in df.groupby(by='n_cluster'):
#     (a,b) = cluster_dat[['num_transformer_blocks','mlp_units']].iloc[cluster_dat['RMSE'].argmin()]
#     l_u.append((a,b))










