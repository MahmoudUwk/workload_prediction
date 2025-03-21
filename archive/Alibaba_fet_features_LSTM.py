
import pandas as pd
import numpy as np
import pickle
import os
from Alibaba_helper_functions import loadDatasetObj,save_object
        
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
    
# 
# base_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/"
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

def df_from_M_id(df,M):
    return df.loc[df[" machine id"].isin(M)]
def get_dataset_alibaba_lstm(seq_length,cluster_num,num_feat=6):
    base_path = "data/"
    
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
    clusters = 0
    if cluster_num !='all':
        clus_obj = 'TimeSeriesKMeans4.obj'
        M_ids_clustered = loadDatasetObj(os.path.join("data/features_lstm",clus_obj))
        # print([len(inds) for inds in M_ids_clustered])
        clusters = [len(inds) for inds in M_ids_clustered]#len(M_ids_clustered)
        M_ids_clustered = M_ids_clustered[cluster_num]
        df = df_from_M_id(df,M_ids_clustered)

    df_normalized = df.copy()
    #%%
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.preprocessing import StandardScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # if normalization == 0:
    #     scaler = StandardScaler()
    # else:
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    
    # scaler.fit(np.array(df[cols]))
    # df_normalized[cols] = scaler.transform(np.array(df[cols]))
    scaler = 100
    df_normalized[cols] = df[cols]/scaler
    del df
    grouped = df_normalized.groupby([" machine id"])
    
    #%%
    train_per_and_val = 0.8
    val_per = 0.3*train_per_and_val
    train_per = train_per_and_val - val_per
    sav_path2 = base_path+"features_lstm/"+"LSTM_M_id_features"
    if not os.path.exists(sav_path2):
        os.makedirs(sav_path2)
        
    X_train_all = []
    y_train_all = []
    
    X_val_all = []
    y_val_all = []
    
    X_test_all = []
    y_test_all = []
    for M_id, M_id_val in grouped:
        # print(M_id)
        M_id_val = M_id_val.sort_values(by=[' timestamp']).reset_index(drop=True).drop([" machine id"," timestamp"],axis=1)[cols]
        
        len_data = M_id_val.shape[0]
        train_len = int(train_per*len_data)
        val_len = int(val_per*len_data)
        target_col_num = [ind for ind,col in enumerate(list(M_id_val.columns)) if col==target][0]
        
        # print(len(M_id_val))
        
        X_train,y_train = sliding_windows2d_lstm(np.array(M_id_val.iloc[:train_len,:]), seq_length,target_col_num)
        X_train_all.append(X_train)
        y_train_all.append(y_train)
        
        if val_per!=0:
            X_val,y_val = sliding_windows2d_lstm(np.array(M_id_val.iloc[train_len:train_len+val_len,:]), seq_length,target_col_num)
            X_val_all.append(X_val)
            y_val_all.append(y_val)   
        
        
        X_test,y_test= sliding_windows2d_lstm(np.array(M_id_val.iloc[train_len+val_len:,:]), seq_length,target_col_num)
        X_test_all.append(X_test)
        y_test_all.append(y_test)   
    
        # dict_Mid = {"X_train":X_train,"y_train":y_train,
        #             "X_val":X_val,"y_val":y_val,
        #             "X_test":X_test,"y_test":y_test}
        # assert(len(y_train)==len(X_train))
        # save_object(dict_Mid, os.path.join(sav_path2,'X_Y_LSTM_M_id_'+str(M_id)+'.obj'))
    

    return list_to_array(X_train_all,seq_length,len(cols)),list_to_array(y_train_all,0,len(cols)),list_to_array(X_val_all,seq_length,len(cols)),list_to_array(y_val_all,0,len(cols)),list_to_array(X_test_all,seq_length,len(cols)),list_to_array(y_test_all,0,len(cols)),scaler,clusters







