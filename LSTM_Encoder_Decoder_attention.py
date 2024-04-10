# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:26:08 2024

@author: mahmo
"""

# import pandas as pd
import numpy as np
from keras.layers import  LSTM,Dense#,Bidirectional
from keras.layers import TimeDistributed,Flatten
from keras.layers import RepeatVector
from keras.models import  Sequential #,load_model
from keras.callbacks import EarlyStopping
# from attention_decoder import AttentionDecoder
from Alibaba_fet_features_LSTM import get_dataset_alibaba_lstm
def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))
def RMSE(test,pred):
    return np.sqrt(np.mean((np.squeeze(test) - np.squeeze(pred))**2))

def MAE(test,pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test)))

def MAPE(test,pred):
    ind = np.where(test!=0)[0].flatten()
    return 100*np.mean(np.abs(np.squeeze(pred[ind]) - np.squeeze(test[ind]))/np.abs(np.squeeze(test[ind])))

seq_length = 7
cluster_num = 'all'
X_train,y_train,X_val,y_val,X_test ,y_test,scaler,clusters = get_dataset_alibaba_lstm(seq_length,cluster_num)
ind_rand = np.random.permutation(len(X_train))
X_train = X_train[ind_rand]
y_train = y_train[ind_rand]
y_test = y_test*scaler
input_dim=(X_train.shape[1],X_train.shape[2])
print(X_train.shape)
output_dim = 1
print(input_dim,output_dim)
y_train = expand_dims(expand_dims(y_train))
y_val = expand_dims(expand_dims(y_val))
if len(X_train.shape)<3:
    X_train = expand_dims(X_train)
    X_val = expand_dims(X_val)
    X_test = expand_dims(X_test)
print(X_train.shape,y_train.shape,X_test.shape,X_val.shape)

n_units = 128
model = Sequential()
model.add(LSTM(n_units, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(n_units, return_sequences=True))
model.add(LSTM(n_units, return_sequences=True))
model.add(LSTM(n_units, return_sequences=True))
# model.add(LSTM(n_units, return_sequences=True))
# model.add(LSTM(n_units, return_sequences=True))
model.add(TimeDistributed(Dense(n_units,activation='relu')))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
print(model.summary())
#%%
epochs_num = 1000
batch_size_n = 2**10
val_split_size = 0.3
callbacks_list = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
if X_val.size ==0:
    history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_split=val_split_size,callbacks=callbacks_list)
else:
    print("val provided")
    history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
                            
# history = model.fit(X_train, y_train, epochs=1000, batch_size=2**10, verbose=2, shuffle=True, validation_split=0.3,callbacks=callbacks_list)

#%%
y_test_pred = model.predict(X_test)*scaler



rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)

base_path = "data/"
sav_path = base_path+"results/lstm"

print(rmse,mae,mape)
def log_results_LSTM(row,save_path):
    import pandas as pd
    import os
    save_name = 'encoder.csv'
    cols = ["loss", "RMSE", "MAE", "MAPE(%)","seq","num_layers","units","best epoch","train_time(min)","n_cluster"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)

row = ['encoder',rmse,mae,mape,seq_length,0,n_units,0,0,0]
 
log_results_LSTM(row,sav_path)