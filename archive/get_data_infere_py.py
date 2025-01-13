
def loadDatasetObj(fname):
    import pickle
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    return data_dict
import numpy as np

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

def process_data_LSTM(group,cols,target,seq_length):
    X = []
    Y = []
    M_ids = []
    target = 'cpu_utilization'
    id_m = "machine_id"
    sort_by = 'start_time'
    for M_id, M_id_values in group:
        M_id_values = M_id_values.sort_values(by=[sort_by]).reset_index(drop=True).drop([id_m,sort_by],axis=1)[cols]
        
        target_col_num = [ind for ind,col in enumerate(list(M_id_values.columns)) if col==target][0]
    
        X_train,y_train = sliding_windows2d_lstm(np.array(M_id_values), seq_length,target_col_num)
        X.append(X_train)
        Y.append(y_train)
        M_ids.append(M_id)
  
    return X,Y,M_ids

def get_data_inf(seq_length,feat_names,target):
    import os
    from args_google import get_paths
    # scaler = 100
    # --- Configuration ---
    base_path, processed_path, feat_google_step1, feat_google_step2, feat_google_step3, sav_path, sav_path_plot = get_paths()
    target = 'cpu_utilization'
    id_m = "machine_id"
    sort_by = 'start_time'
    # -----------------------
    # --- Load Data ---
    cols = feat_names+[id_m]+[sort_by]
    df = loadDatasetObj(os.path.join(base_path, 'google.obj'))[cols]
    # k_threshold = 0.3
    # --- Filter Machines based on Average CPU Usage ---
    # average_values = df.groupby(id_m)[target].mean()
    # selected_machines = average_values[average_values > k_threshold].index.tolist()
    filename = os.path.join(sav_path,'Adaptive_predictor.obj')
    selected_machines = loadDatasetObj(filename)['Mids_test']
    df = df[df[id_m].isin(selected_machines)]
    df = df.dropna()
    X,Y,_ = process_data_LSTM(df.groupby(id_m),feat_names,target,seq_length)

    return X,Y,selected_machines
