# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:10:02 2024

@author: msallam
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle

def get_data_stat(data_path):
    from sklearn.preprocessing import MinMaxScaler
    df = loadDatasetObj(data_path)


    X_train = np.array(df['XY_train'].drop(['M_id','y'],axis=1))

    y_train = np.array(df['XY_train']['y'])


    X_test = np.array(df['XY_test'].drop(['M_id','y'],axis=1))


    y_test = np.array(df['XY_test']['y'])
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train,y_train,X_test,y_test,scaler,df['XY_test']


def flatten(xss):
    return [x for xs in xss for x in xs]

def get_EN_DE_LSTM_model(input_dim,output_dim,units,num_layers,dense_units,seq=0,lr=0,name='LSTM_HP'):
    from keras.layers import Dense,LSTM,RepeatVector,TimeDistributed,Flatten
    from keras.models import Sequential
    from keras.optimizers import Adam
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


def get_train_test_Mids(base_path,train_val_per,val_per):
    # base_path = "data/"
    script = "server_usage.csv"
    target = " used percent of cpus(%)"

    
    info_path = base_path+"schema.csv"
    
    df_info =  pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']
    
    
    full_path = base_path+script
    nrows = None
    df =  pd.read_csv(full_path,nrows=nrows,header=None,names=list(df_info))

    M_ids = np.array(list(df[" machine id"].unique()))
    np.random.seed(8)
    indeces_rearrange_random = np.random.permutation(len(M_ids))
    M_ids = M_ids[indeces_rearrange_random]
    
    train_val_per = 0.8
    train_per = train_val_per - val_per
    
    train_len  = int(train_per * len(M_ids))
    val_len = int(val_per * len(M_ids))
    return M_ids[:train_len],M_ids[train_len:train_len+val_len],M_ids[train_len+val_len:]

def drop_col_nan_inf(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(axis=1)


def list_to_array(lst,seq_length,n_feat):
    shapes = 0
    for sub_list in lst:
        shapes += len(sub_list)
    ind = 0
    if seq_length != 0:
        X = np.zeros((shapes,seq_length,n_feat))
        for sub_list in lst:
            X[ind:ind+len(sub_list),:,:] = sub_list
            ind = ind + len(sub_list)
    else:
        X = np.zeros((shapes,))
        for sub_list in lst:
            X[ind:ind+len(sub_list)] = sub_list
            ind = ind + len(sub_list)
  
    return X


def diff(test,pred):
    return np.abs(test-pred)


def expand_dims_st(X):
    return np.expand_dims(X, axis = 0)


def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))


def RMSE(test,pred):
    return np.sqrt(np.mean((np.squeeze(np.array(test)) - np.squeeze(np.array(pred)))**2))

def MAE(test,pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test)))

def MAPE(test,pred):
    test = np.array(test)
    pred = np.array(pred)
    ind = np.where(test!=0)[0].flatten()
    return 100*np.mean(np.abs(np.squeeze(pred[ind]) - np.squeeze(test[ind]))/np.abs(np.squeeze(test[ind])))


def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def get_Mid():
    script = "server_usage.csv"
    target = " used percent of cpus(%)"
    
    base_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/"
    # base_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/"
    info_path = base_path+"schema.csv"
    
    df_info =  pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']
    
    
    full_path = base_path+script
    nrows = None
    df =  pd.read_csv(full_path,nrows=nrows,header=None,names=list(df_info))
    
    df = df[[" machine id", " timestamp"," used percent of cpus(%)"]]
    # df = df[df.notna()]
    df = df.dropna()
    
    
    
    df_grouped_id = df.groupby([" machine id"])
    
    mean_Mid= np.expand_dims(np.array(df_grouped_id.mean()[target]), axis=1)
    std_Mid= np.expand_dims(np.array(df_grouped_id.std()[target]), axis=1)
    X_Mid = np.concatenate((mean_Mid,std_Mid),axis=1)
    
    
    
    
    
    kmeans = KMeans(n_clusters=4, n_init="auto",algorithm='elkan',max_iter=5000,random_state=5).fit(X_Mid)
    
    M_id_labels = kmeans.labels_
    
    return [np.where(M_id_labels==class_Mid)[0] for  class_Mid in np.unique(kmeans.labels_)]
