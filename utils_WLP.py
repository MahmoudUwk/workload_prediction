import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
#%%
def flatten(xss):
    return [x for xs in xss for x in xs]

def drop_col_nan_inf(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(axis=1)


def list_to_array(lst):
    if lst[0].ndim==3:
        _,seq_length,n_feat = lst[0].shape
        flag = 0
    else:
        flag = 1
    shapes = 0
    for sub_list in lst:
        shapes += len(sub_list)
    ind = 0
    if flag == 0:
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

def split_3d_array(array_3d, batch_size):
    num_samples = array_3d.shape[0]
    sub_arrays = []
    for i in range(0, num_samples, batch_size):
        sub_array = array_3d[i:i + batch_size]
        sub_arrays.append(sub_array)
    return sub_arrays

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

def df_from_M_id(df,M):
    return df.loc[df["id"].isin(M)]

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
        
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
    
def window_rolling(df,seq_length):
    cols = ['y','x1']
    target = 'y'
    X = []
    Y = []
    M_ids = []
    for M_id, M_id_values in df.groupby(["id"]):
        M_id_values = M_id_values.sort_values(by=['date']
                  ).reset_index(drop=True).drop(["id","date"],axis=1)[cols]
        
        target_col_num = [ind for ind,col in enumerate(list(M_id_values.columns)) if col==target][0]
    
        X_train,y_train = sliding_window_i(np.array(M_id_values), seq_length,target_col_num)
        X.append(X_train)
        Y.append(y_train)
        M_ids.append(M_id)
  
    return X,Y,M_ids

#%% LSTM models


def get_en_de_lstm_model_attention(input_shape,output_dim, units, num_layers,dense_units, lr=0.005, seq=12,
       name='EncoderDecoderLSTM_MTO_Attention'):
    from keras.layers import Dense,Flatten, LSTM, Input, AdditiveAttention, concatenate, RepeatVector
    from keras.models import Model
    from keras.optimizers import Adam
    # Encoder
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    encoder_outputs = LSTM(units, return_sequences=True)(encoder_inputs)
    for i in range(num_layers - 2):
        encoder_outputs = LSTM(units, return_sequences=True, name=f"encoder_lstm_{i+1}")(encoder_outputs)
    
    # Final encoder layer returns both sequences and states
    encoder_outputs, state_h, state_c = LSTM(units, return_sequences=True, return_state=True, name=f"encoder_lstm_{num_layers-1}")(encoder_outputs)
    encoder_states = [state_h, state_c]

    # Decoder
    # RepeatVector to expand the final encoder state to match the expected decoder input
    context = RepeatVector(1, name="context_vector")(encoder_outputs[:, -1, :])

    # Decoder layers
    decoder = LSTM(units, return_sequences=True, name="decoder_lstm_0")(context, initial_state=encoder_states)
    for i in range(num_layers - 1):
        decoder = LSTM(units, return_sequences=True, name=f"decoder_lstm_{i+1}")(decoder)

    # Attention Mechanism
    attention_layer = AdditiveAttention(name="attention_layer")
    attention_output = attention_layer([decoder, encoder_outputs])

    # Concatenate attention output and decoder output
    decoder_combined_context = concatenate([decoder, attention_output], axis=-1, name="concatenate_output_attention")

    # Dense layers for final prediction
    outputs = Dense(dense_units, activation='relu', name="dense_1")(decoder_combined_context)
    outputs = Dense(output_dim, activation=None, name="output_layer")(outputs)

    # Flatten the output
    outputs = Flatten(name="flatten_output")(outputs)

    # Create and compile model
    model = Model(inputs=encoder_inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model


def get_lstm_model(input_shape, output_dim, units, num_layers, dense_units, lr=0.001, seq=12, name='LSTM_Model'):
    from keras.layers import Dense, LSTM, Input
    from keras.models import Model
    from keras.optimizers import Adam

    # Input Layer
    inputs = Input(shape=input_shape, name='input')

    # LSTM Layers
    lstm_outputs = inputs  # Initialize with the input
    for i in range(num_layers):
        lstm_outputs = LSTM(units, return_sequences=(i < num_layers - 1),
                           name=f"lstm_{i}")(lstm_outputs)

    # Dense Layers
    outputs = Dense(dense_units, activation='relu', name="dense_1")(lstm_outputs)
    outputs = Dense(output_dim, activation=None, name="output_layer")(outputs)

    # Model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

#%% Alibaba dataset adaptive preprocessing

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


#%% Alibaba dataset split ids train val test

def get_train_test_Mids(df,train_val_per,val_per):
    np.random.seed(8)
    M_ids = np.array(list(df["id"].unique()))
    indeces_rearrange_random = np.random.permutation(len(M_ids))
    M_ids = M_ids[indeces_rearrange_random]

    train_per = train_val_per - val_per
    
    train_len  = int(train_per * len(M_ids))
    val_len = int(val_per * len(M_ids))
    return M_ids[:train_len],M_ids[train_len:train_len+val_len],M_ids[train_len+val_len:]



# 1. Data Loading and Preprocessing Using get_df_tft
def prepare_data_for_tft(df,rename_map):
    

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
    df_info = pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']
    full_path = base_path + script
    nrows = None
    df = pd.read_csv(
        full_path, nrows=nrows, header=None, names=list(df_info)
    )
    
    df.dropna(inplace=True)
    normalize_cols = ['y', 'x1']
    rename_map = {
        ' used percent of cpus(%)': "y",
        ' used percent of memory(%)': "x1",
        ' machine id': "id",
        ' timestamp':'date',
    }
    df.rename(
        columns=rename_map,
        inplace=True,
    )
    df = df.loc[:,list(rename_map.values())]
    scaler = 100
    df.loc[:, normalize_cols] = df.loc[:, normalize_cols] / scaler
    
    reference_time = datetime(2017, 1, 1)
    df["date"] = df["date"].apply(lambda x: reference_time + timedelta(seconds=x))
    
    df = df.sort_values(['id', 'date'])
    

    df.reset_index(drop=True, inplace=True)
    
    train_val_per = 0.8
    val_per = 0.16

    M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(
        df,train_val_per ,val_per)

    df_train = df_from_M_id(df, M_ids_train)
    df_val = df_from_M_id(df, M_ids_val)
    df_test = df_from_M_id(df, M_ids_test)

    return df_train, df_val, df_test
#%% google get dfs
def get_df_google():
    from args_google import get_paths
    base_path, _, _, _, _, sav_path, _ = get_paths()
    target = 'cpu_utilization'
    id_m = "machine_id"
    sort_by = 'start_time'

    df = loadDatasetObj(os.path.join(base_path, 'google.obj'))

    filename = os.path.join(sav_path,'Adaptive_predictor.obj')
    selected_machines = loadDatasetObj(filename)['Mids_test']
    df = df[df[id_m].isin(selected_machines)]

    df.dropna(inplace=True)
    normalize_cols = ['y', 'x1']
    rename_map = {
        target: "y",
        'memory_utilization': "x1",
        id_m: "id",
        sort_by:'date',
    }
    df.rename(
        columns=rename_map,
        inplace=True,
    )
    df = df.loc[:,list(rename_map.values())]
    scaler = 100
    df.loc[:, normalize_cols] = df.loc[:, normalize_cols] / scaler
    #2020-04-01,
    offset_seconds = 600
    df['date'] = pd.to_datetime(df['date'] / 1e6 + offset_seconds, unit='s')
    start_date = pd.to_datetime('2019-05-01')
    df['date'] = start_date + (df['date'] - pd.to_datetime('1970-01-01'))

    df = df.sort_values(['id', 'date'])
    

    df.reset_index(drop=True, inplace=True)
    
    train_val_per = 0.8
    val_per = 0.16

    M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(
        df,train_val_per ,val_per)

    df_train = df_from_M_id(df, M_ids_train)
    df_val = df_from_M_id(df, M_ids_val)
    df_test = df_from_M_id(df, M_ids_test)

    return df_train, df_val, df_test
#%% bitbrain get df
def get_df_BB():
    import os
    from args_BB import get_paths
    # scaler = 100
    # --- Configuration ---
    base_path, processed_path, feat_BB_step1, feat_BB_step2, feat_BB_step3, sav_path, sav_path_plot = get_paths()
    target = 'CPU usage [%]'
    id_m = "machine_id"
    sort_by = 'date'
    # -----------------------
    # --- Load Data ---

    df = loadDatasetObj(os.path.join(base_path, 'rnd.obj'))
    df['memory_utilization'] = (df['Memory usage [KB]'] / df['Memory capacity provisioned [KB]']) * 100
    df.loc[df['Memory capacity provisioned [KB]'] == 0, 'memory_utilization'] = 0
 
    df['memory_utilization'] = df['memory_utilization'].clip(lower=0, upper=100)
    filename = os.path.join(sav_path,'Adaptive_predictor.obj')
    selected_machines = loadDatasetObj(filename)['Mids_test']
    df = df[df[id_m].isin(selected_machines)]
    df = df.dropna()

    
    df.dropna(inplace=True)
    normalize_cols = ['y', 'x1']
    rename_map = {
        target: "y",
        'memory_utilization': "x1",
        id_m: "id",
        sort_by:'date',
    }
    df.rename(
        columns=rename_map,
        inplace=True,
    )
    df = df.loc[:,list(rename_map.values())]
    scaler = 100
    df.loc[:, normalize_cols] = df.loc[:, normalize_cols] / scaler

    # reference_time = datetime(2013, 8, 1)
    # df['date'] = pd.to_datetime(df['date'], unit='ms')

    
    df = df.sort_values(['id', 'date'])
    

    df.reset_index(drop=True, inplace=True)
    
    train_val_per = 0.8
    val_per = 0.16

    M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(
        df,train_val_per ,val_per)

    df_train = df_from_M_id(df, M_ids_train)
    df_val = df_from_M_id(df, M_ids_val)
    df_test = df_from_M_id(df, M_ids_test)

    return df_train, df_val, df_test



#%% main function to use get_dicts
def get_dict_df(df,seq_length):
    X_list,Y_list,M_ids = window_rolling(df,seq_length)
    dict_df = {'X_list':X_list,'Y_list':Y_list,'M_ids':M_ids,
               'X':list_to_array(X_list),'Y':list_to_array(Y_list)}
    return dict_df

def get_dicts(df_train, df_val, df_test,seq_length):
    train_dict = get_dict_df(df_train,seq_length)
    val_dict = get_dict_df(df_val,seq_length)
    test_dict = get_dict_df(df_test,seq_length)
    return train_dict , val_dict, test_dict

def get_dict_option(flag,seq_length):
    if flag == "Alibaba":
        df_train, df_val, df_test = get_df_alibaba()
    elif flag == "google":
        df_train, df_val, df_test = get_df_google()
    elif flag == "BB":
        df_train, df_val, df_test = get_df_BB()
        
    train_dict , val_dict, test_dict = get_dicts(df_train, df_val, 
                                                 df_test,seq_length)
    return train_dict , val_dict, test_dict

        

#%%
def model_serv(X_list,Y_list,model,scaler,batch_size):
    y_test_pred_list = []
    rmse_list = []
    for c,test_sample_all in enumerate(X_list):
        if len(test_sample_all)>batch_size:
            test_sample_all = split_3d_array(test_sample_all, batch_size)
        else:
            test_sample_all = [test_sample_all]
        pred_i = []
        for test_sample in test_sample_all:
            pred_ii = list(np.squeeze(np.array(model.predict(test_sample))) *scaler)
            pred_i.append(pred_ii)
        pred_i = flatten(pred_i)
        y_test_pred_list.append(pred_i)
        rmse_i_list = RMSE(Y_list[c]*scaler,pred_i)
        rmse_list.append(rmse_i_list)
    return y_test_pred_list,rmse_list

#%%
def log_results_EB0(row,cols,save_name):
    if not os.path.isfile(save_name):
        df3 = pd.DataFrame(columns=cols)
        df3.to_csv(save_name,index=False)   
    df = pd.read_csv(save_name)
    df.loc[len(df)] = row
    df.to_csv(save_name,mode='w', index=False,header=True)










