import pandas as pd
import numpy as np
import pickle
import os
from Alibaba_helper_functions import loadDatasetObj,save_object,list_to_array,get_train_test_Mids
        
def sliding_windows2d_lstm(data, seq_length,target_col_num):
    
    x = np.zeros((len(data)-seq_length,seq_length,data.shape[1]))
    y = np.zeros((len(data)-seq_length))
    #print(x.shape,y.shape)
    for ind in range(len(x)):
        #print((i,(i+seq_length)))
        x[ind,:,:] = data[ind:ind+seq_length,:]
        #print(data[ind+seq_length:ind+seq_length+k_step])
        y[ind] = data[ind+seq_length:ind+seq_length+1,target_col_num][0]
    return x,y
    

def df_from_M_id(df,M):
    return df.loc[df[" machine id"].isin(M)]

def process_data_LSTM(group,cols,target,seq_length):
    X = []
    Y = []
    M_ids = []
    for M_id, M_id_values in group:
        M_id_values = M_id_values.sort_values(by=[' timestamp']).reset_index(drop=True).drop([" machine id"," timestamp"],axis=1)[cols]
        
        target_col_num = [ind for ind,col in enumerate(list(M_id_values.columns)) if col==target][0]
    
        X_train,y_train = sliding_windows2d_lstm(np.array(M_id_values), seq_length,target_col_num)
        X.append(X_train)
        Y.append(y_train)
        M_ids.append(M_id)
  
    return X,Y,M_ids


def get_dataset_alibaba_lstm_no_cluster(seq_length,num_feat=6):
    base_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/'
    # base_path = "data/"
    
    sav_path = base_path+"features_lstm"
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
        
        
    script = "server_usage.csv"
    target = " used percent of cpus(%)"
    cols = [' used percent of cpus(%)',
            ' used percent of memory(%)', ' used percent of disk space(%)',
            ' linux cpu load average of 1 minute',
            ' linux cpu load average of 5 minute',
            ' linux cpu load average of 15 minute'][:num_feat]
    # cols = [' used percent of cpus(%)']
    pd.set_option('display.expand_frame_repr', False)
    pd.options.display.max_columns = None
    
    info_path = base_path+"schema.csv"
    
    df_info =  pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']
    #%%
    full_path = base_path+script
    nrows = None
    df =  pd.read_csv(full_path,nrows=nrows,header=None,names=list(df_info))
    df = df.dropna()
    
    train_val_per = 0.8
    val_per = 0.16

    M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(base_path,train_val_per,val_per)


    df_train = df_from_M_id(df,M_ids_train)
    df_val = df_from_M_id(df,M_ids_val)
    df_test = df_from_M_id(df,M_ids_test)

    scaler = 100
    df_train[cols] = df_train[cols]/scaler
    df_val[cols] = df_val[cols]/scaler
    df_test[cols] = df_test[cols]/scaler

    
    #%%

    sav_path2 = base_path+"features_lstm/"+"LSTM_M_id_features"
    if not os.path.exists(sav_path2):
        os.makedirs(sav_path2)
        
    X_train_all,y_train_all,_ = process_data_LSTM(df_train.groupby([" machine id"]),cols,target,seq_length)
    
    if val_per!=0:
        X_val_all,y_val_all,_ = process_data_LSTM(df_val.groupby([" machine id"]),cols,target,seq_length)
    
    X_test_all,y_test_all,Mids_test = process_data_LSTM(df_test.groupby([" machine id"]),cols,target,seq_length)


    return list_to_array(X_train_all,seq_length,len(cols)),list_to_array(y_train_all,0,len(cols)),list_to_array(X_val_all,seq_length,len(cols)),list_to_array(y_val_all,0,len(cols)),X_test_all,y_test_all,scaler,Mids_test







