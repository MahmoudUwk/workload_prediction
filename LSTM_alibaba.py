from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from keras.layers import  Bidirectional,LSTM, BatchNormalization,Dense#,Bidirectional
from keras.models import  Sequential #,load_model
from keras.optimizers import Adam,RMSprop,Adadelta
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
import keras
from keras.callbacks import EarlyStopping
import pickle
# from Alibaba_helper_functions import get_Mid
from Alibaba_fet_features_LSTM import get_dataset_alibaba_lstm

#%%
columns=['Steps', 'LSTM Units', 'RMSE','NRMSE', 'Best Epoch', 'Num epochs','seq_length']

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))

def inverse_transf(X,scaler):
    if 'var_' in list(scaler.__dict__.keys()):
        return (np.sqrt(scaler.var_[0]) * X )+ scaler.mean_[0]
    return np.array((X *(scaler.data_max_[0]-scaler.data_min_[0]) )+scaler.data_min_[0])

def RMSE(test,pred):
    return np.sqrt(np.mean((np.squeeze(test) - np.squeeze(pred))**2))

def MAE(test,pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test)))

def MAPE(test,pred):
    ind = np.where(test!=0)[0].flatten()
    return 100*np.mean(np.abs(np.squeeze(pred[ind]) - np.squeeze(test[ind]))/np.abs(np.squeeze(test[ind])))


def get_LSTM_model(input_dim,output_dim,units,num_layers,name='LSTM_HP'):
    model = Sequential(name=name)
    flag_seq = True
    if num_layers == 1:
        model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = False))
    else:
        model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True))
    for dummy in range(num_layers-1):
        if dummy == num_layers-2:
            flag_seq = False     
        model.add(LSTM(units=units,return_sequences = flag_seq))
    model.add(Dense(output_dim))
    return model

def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict



def log_results_LSTM(row,save_path):

    save_name = 'results_LSTM_val2.csv'
    cols = ["loss", "RMSE", "MAE", "MAPE(%)","seq","num_layers","units","best epoch","train_time(min)","norm_type"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    
#%%
base_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/"

sav_path = base_path+"results/lstm"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

alg_name = 'LSTM'
lr = 0.00002
opt_chosen=Adam(learning_rate=lr)




epochs_num = 1000
drop_out = 0
callback_falg = 1

output_dim = 1
batch_size_n = 2**9


num_units = [12,16,32,64]#[35]#[8,10,15]
num_layers_all = [3,4,5]

norm_all = [0,1]
norm_names = ['Standard','MinMax']
losses = ['mse','mae']
seq_length = 12
for cluster_num  in range(4):
    for norm_i in norm_all:
        X_train,y_train,X_val,y_val,X_test ,y_test,scaler = get_dataset_alibaba_lstm(seq_length,norm_i,cluster_num)
    
        ind_rand = np.random.permutation(len(X_train))
        X_train = X_train[ind_rand]
        y_train = y_train[ind_rand]
        y_test = inverse_transf(y_test,scaler)
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
        y_test = expand_dims(y_test)
        for loss in losses:
            for num_layers in num_layers_all:   
                for units in num_units:
                    
                    model = get_LSTM_model(input_dim,output_dim,units,num_layers)         
                    model.compile(optimizer=opt_chosen, loss=loss)
                    print(model.summary())
                    print('start training')
                    start_train = time.time()
                    if callback_falg:
                        checkpoint = SaveBestModel()
                        # callbacks_list = [checkpoint]
                        callback_es = EarlyStopping(monitor='val_loss', patience=10)
                        callbacks_list = [checkpoint,callback_es]
                        history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
                    else:
                        history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True)#, validation_split=0.2,callbacks=callbacks_list)
                    #%%
                    end_train = time.time()
                    print('End training')
                    train_time = (end_train - start_train)/60
                    if callback_falg:
                        model.set_weights(checkpoint.best_weights)
                        best_epoch =np.argmin(history.history['val_loss'])
                    else:
                        best_epoch = 0
                    # model.save(filepath)
                    #%%
                    
                    start_test = time.time()
                    y_test_pred = inverse_transf(model.predict(X_test),scaler)
                    end_test = time.time()
                    test_time = end_test - start_test
                    #%%
                    rmse = RMSE(y_test,y_test_pred)
                    mae = MAE(y_test,y_test_pred)
                    mape = MAPE(y_test,y_test_pred)
                    print(rmse,mae,mape)
                    # best_epoch = epochs_num
                    
                    row = [loss,rmse,mae,mape,seq_length,num_layers,units,best_epoch,train_time,norm_names[norm_i]]
             
                    log_results_LSTM(row,sav_path)
                    #%%
                    # plt.figure(figsize=(10,7),dpi=180)
                    # plt.plot(test_time_axis,1000*np.squeeze(y_test), color = 'red', linewidth=2.0, alpha = 0.6)
                    # plt.plot(test_time_axis,1000*np.squeeze(y_test_pred), color = 'blue', linewidth=0.8)
                    # plt.legend(['Actual','Predicted'])
                    # plt.xlabel('Timestamp')
                    # plt.xticks( rotation=25)
                    # plt.ylabel('mW')
                    # plt.title('Energy Prediction using '+alg_name)
                    # plt.show()
                    # info_loop = [seq,num_layers,units,best_epoch,datatype_opt]
                    # name_sav = ""
                    # for n in info_loop:
                    #     name_sav = name_sav+str(n)+"_" 
                    # plt.savefig(os.path.join(save_path,'LSTM'+name_sav+'.png'))
                    # plt.close()
                    # filename = os.path.join(save_path,alg_name+'.obj')
                    # obj = {'y_test':y_test,'y_test_pred':y_test_pred}
                    # save_object(obj, filename)
