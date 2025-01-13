import pandas as pd
import numpy as np
import pickle
import os
from Alibaba_helper_functions import loadDatasetObj,save_object,list_to_array,get_train_test_Mids
        
def sliding_window_i(data, seq_length,target_col_num):
    
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

def window_rolling(group,cols,target,seq_length):
    X = []
    Y = []
    M_ids = []
    for M_id, M_id_values in group:
        M_id_values = M_id_values.sort_values(by=[' timestamp']).reset_index(drop=True).drop([" machine id"," timestamp"],axis=1)[cols]
        
        target_col_num = [ind for ind,col in enumerate(list(M_id_values.columns)) if col==target][0]
    
        X_train,y_train = sliding_window_i(np.array(M_id_values), seq_length,target_col_num)
        X.append(X_train)
        Y.append(y_train)
        M_ids.append(M_id)
  
    return X,Y,M_ids

#%%
def get_dataset_alibaba_lstm_no_cluster(seq_length,num_feat=6):
    from args import get_paths
    base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()

    # base_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/'
    # base_path = "data/"
    
    sav_path = base_path+"features_lstm"
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
        
        
    script = "server_usage.csv"
    target = " used percent of cpus(%)"
    cols = [' used percent of cpus(%)',' used percent of memory(%)']
    # cols = [' used percent of cpus(%)']
    pd.set_option('display.expand_frame_repr', False)
    pd.options.display.max_columns = None
    
    info_path = base_path+"schema.csv"
    
    df_info =  pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']
   
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
    df_train.loc[:, cols] = df_train.loc[:, cols] / scaler
    df_val.loc[:, cols] = df_val.loc[:, cols] / scaler
    df_test.loc[:, cols] = df_test.loc[:, cols] / scaler

        
    X_train_all,y_train_all,_ = window_rolling(df_train.groupby([" machine id"]),cols,target,seq_length)
    
    if val_per!=0:
        X_val_all,y_val_all,_ = window_rolling(df_val.groupby([" machine id"]),cols,target,seq_length)
    
    X_test_all,y_test_all,Mids_test = window_rolling(df_test.groupby([" machine id"]),cols,target,seq_length)


    return list_to_array(X_train_all,seq_length,len(cols)),list_to_array(y_train_all,0,len(cols)),list_to_array(X_val_all,seq_length,len(cols)),list_to_array(y_val_all,0,len(cols)),X_test_all,y_test_all,scaler,Mids_test


#%%

# 1. Data Loading and Preprocessing Using get_df_tft
def prepare_data_for_tft(df,rename_map):
    from datetime import datetime, timedelta

    reference_time = datetime(2017, 1, 1)
    df[" timestamp"] = df[" timestamp"].apply(lambda x: reference_time + timedelta(seconds=x))
    
    df = df.sort_values([' machine id', ' timestamp'])
    
    df.dropna(inplace=True)
    df.rename(
        columns=rename_map,
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df
def get_df_alibaba():
    from args import get_paths

    base_path, _, _, _, _, _, _ = get_paths()

    script = "server_usage.csv"
    info_path = base_path + "schema.csv"
    normalize_cols = [' used percent of cpus(%)', ' used percent of memory(%)']
    cols = [
        ' timestamp',
        ' used percent of cpus(%)',
        ' used percent of memory(%)',
        ' machine id',
    ]
    rename_map = {
        ' used percent of cpus(%)': "y",
        ' used percent of memory(%)': "mem_util",
        ' machine id': "unique_id",
        ' timestamp':'ds',
    }
    df_info = pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']

    full_path = base_path + script
    nrows = None
    df = pd.read_csv(
        full_path, nrows=nrows, header=None, names=list(df_info)
    )[cols]
    scaler = 100
    df.loc[:, normalize_cols] = df.loc[:, normalize_cols] / scaler
    df = df.dropna()

    train_val_per = 0.8
    val_per = 0.16

    M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(
        base_path, train_val_per, val_per
    )

    df_train = prepare_data_for_tft(df_from_M_id(df, M_ids_train),rename_map)
    df_val = prepare_data_for_tft(df_from_M_id(df, M_ids_val),rename_map)
    df_test = prepare_data_for_tft(df_from_M_id(df, M_ids_test),rename_map)

    return df_train, df_val, df_test,"cpu_util"
#%%


def get_train_val_test(seq_length,target,cols,):
    from args import get_paths
    base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()

    # base_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/'
    # base_path = "data/"
    
    sav_path = base_path+"features_lstm"
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
        
        
    script = "server_usage.csv"
    target = " used percent of cpus(%)"
    cols = [' used percent of cpus(%)',' used percent of memory(%)']

    
    info_path = base_path+"schema.csv"
    
    df_info =  pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']
   
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
    df_train.loc[:, cols] = df_train.loc[:, cols] / scaler
    df_val.loc[:, cols] = df_val.loc[:, cols] / scaler
    df_test.loc[:, cols] = df_test.loc[:, cols] / scaler

        
    X_train_all,y_train_all,_ = window_rolling(df_train.groupby([" machine id"]),cols,target,seq_length)
    

    X_val_all,y_val_all,_ = window_rolling(df_val.groupby([" machine id"]),cols,target,seq_length)
    
    X_test_all,y_test_all,Mids_test = window_rolling(df_test.groupby([" machine id"]),cols,target,seq_length)


    return list_to_array(X_train_all,seq_length,len(cols)),list_to_array(y_train_all,0,len(cols)),list_to_array(X_val_all,seq_length,len(cols)),list_to_array(y_val_all,0,len(cols)),X_test_all,y_test_all,scaler,Mids_test



