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

def get_EN_DE_LSTM_model_old(input_dim,output_dim,units,num_layers,dense_units,seq=12,lr=0.001,name='LSTM_HP'):
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

def get_EN_DE_LSTM_model(input_shape, output_dim, units, num_layers, dense_units,seq=12, lr=0.001, name='EncoderDecoderLSTM'):
    from keras.layers import Dense,LSTM,RepeatVector,TimeDistributed,Input
    from keras.models import Model
    from keras.optimizers import Adam

    # Encoder
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    encoder = encoder_inputs
    for i in range(num_layers):
        encoder = LSTM(units, return_sequences=(i < num_layers - 1), name=f"encoder_lstm_{i}")(encoder)
    # Encoder output is the context vector
    context_vector = encoder

    # Repeat the context vector to match the expected 3D input for the decoder
    repeated_context = RepeatVector(1, name="repeat_context")(context_vector)  # Repeat once for many-to-one

    # Decoder
    decoder = LSTM(units, return_sequences=False, name="decoder_lstm")(repeated_context)

    # Output Layer
    decoder_outputs = Dense(dense_units, activation='relu', name="dense_1")(decoder)
    decoder_outputs = Dense(output_dim, activation=None, name="output_layer")(decoder_outputs)  # Single scalar output

    # Model
    model = Model(inputs=encoder_inputs, outputs=decoder_outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model
#%% Attention

def get_en_de_lstm_model_attention(input_shape,output_dim, units, num_layers,dense_units, lr=0.005, seq=12,
       name='EncoderDecoderLSTM_MTO_Attention'):
    from keras.layers import Dense,Flatten, LSTM, Input, AdditiveAttention, concatenate, RepeatVector
    from keras.models import Model
    from keras.optimizers import Adam
    import tensorflow as tf
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
#%%

# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def get_lstm_model(input_shape, output_dim, units, num_layers, dense_units, lr=0.001, seq=12, name='LSTM_Model'):
    from keras.layers import Dense, LSTM, Input, Dropout
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

#%%
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
    
    # train_val_per = 0.8
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
    from args import get_paths
    base_path,processed_path,_,_,feat_stats_step3,sav_path = get_paths()
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


def get_google_data(seq,scaler):
    from get_data_infere_py import get_data_inf
    feat_names = ['cpu_utilization', 'memory_utilization']
    target = ['cpu_utilization']
    X_list,Y_list,M_ids = get_data_inf(seq,feat_names,target)
    return list_to_array(X_list,seq,len(feat_names))/scaler,list_to_array(Y_list,0,len(feat_names))


def get_BB_data(seq,scaler):
    from get_data_infere_py_BB import get_data_inf_BB
    feat_names = ['CPU usage [%]', 'memory_utilization']
    target = 'CPU usage [%]'
    id_m = "machine_id"
    sort_by = 'Timestamp [ms]'
    X_list,Y_list,M_ids = get_data_inf_BB(seq,feat_names,target)
    return list_to_array(X_list,seq,len(feat_names))/scaler,list_to_array(Y_list,0,len(feat_names))

# from Alibaba_helper_functions import get_google_data, get_BB_data
# X_BB.Y_BB = get_BB_data(seq,scaler)
# X_google.Y_google = get_google_data(seq,scaler)
