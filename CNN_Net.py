import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber


def build_cnn_model(input_shape,num_filter=64, num_classes=1):
    """
    Builds a simple 1D CNN model for time series forecasting.

    Args:
        input_shape: Tuple (seq_len, num_features) representing the input shape.
        num_classes: Number of output classes (1 for regression/forecasting).

    Returns:
        A Keras model.
    """
    # num_filter = 64
    inputs = keras.Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv1D(filters=num_filter, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=num_filter*2, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=num_filter*4, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(pool_size=2)(x)

    # Flatten layer
    x = layers.Flatten()(x)

    # Dense layers
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)  # Optional dropout for regularization

    # Output layer (linear activation for regression)
    outputs = layers.Dense(units=num_classes, activation=None)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
input_shape = (32, 2)  # 32 time steps, 2 features (CPU and memory)
model = build_cnn_model(input_shape, num_classes=1)
model.summary()

# Compile the model
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
#%%
def rmse_tf(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
import os
from keras.callbacks import EarlyStopping
import numpy as np
from utils_WLP import save_object,MAE,MAPE, expand_dims,get_en_de_lstm_model_attention,get_best_lsmt_para
from utils_WLP import get_dict_option,get_lstm_model,model_serv,RMSE,log_results_EB0
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=15, min_lr=1e-4)

scaler = 100
num_epoc = 7000
batch_size = 2**10
patientce_para = 15
lr = 0.0005
seq_len =  32#get_best_lsmt_para(0)['seq']
#swish relu gelu
activation_func = 'swish'
flag_datasets = [2]

data_set_flags = ['Alibaba','google','BB']


num_filters= [64]
cols = ['RMSE','num_stagse','num_blocks','num_filters','kernel_size']

for flag_dataset in flag_datasets:
    if data_set_flags[flag_dataset] == 'Alibaba':
        from args import get_paths
    elif data_set_flags[flag_dataset]=='google':
        from args_google import get_paths
    elif data_set_flags[flag_dataset]=='BB':
        from args_BB import get_paths
    
    base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
    
    train_dict , val_dict, test_dict = get_dict_option(data_set_flags[flag_dataset],seq_len)
    
    
    X_train = train_dict['X']
    y_train = train_dict['Y']
    
    X_val = val_dict['X']
    y_val = val_dict['Y']
    
    X_test_list = test_dict['X_list']
    y_test_list = test_dict['Y_list']
    Mids_test = test_dict['M_ids']
    
    X_test = test_dict['X']
    y_test = test_dict['Y']
    
    output_dim = 1
    input_shape=(X_train.shape[1],X_train.shape[2])
    y_train = expand_dims(expand_dims(y_train))
    y_val = expand_dims(expand_dims(y_val))
    #%%
    RMSE_list = []
    for num_filter in num_filters:
        model = build_cnn_model(input_shape,num_filter)
        
        model.summary()
        model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1)
                      ,metrics=[ 'mape'])
    
        callbacks_list = [EarlyStopping(monitor='val_loss', 
                            patience=patientce_para, restore_best_weights=True),
                          reduce_lr]
        
        start_train = time.time()
        history = model.fit(X_train, y_train, epochs=num_epoc , 
                  batch_size=batch_size, verbose=2, shuffle=True, 
                  validation_data=(X_val,y_val),callbacks=callbacks_list)
        
        end_train = time.time()
        train_time = (end_train - start_train)/60
        
        # X_test,y_test
        y_test_pred = (model.predict(X_test))*scaler
        row_alibaba = [RMSE(y_test*scaler,y_test_pred),MAE(y_test*scaler,y_test_pred)
                       ,MAPE(y_test*scaler,y_test_pred)]
        
        
        
        
        
        # rmse_i = RMSE(y_test,y_test_pred)
        # RMSE_list.append([num_filter,rmse_i])

        # save_name = os.path.join(sav_path,'CNN.csv')
        # ['num_stagse','num_blocks','num_filters','kernel_size']
        # row = [rmse_i,stage,block,filter_i,kss_i]
        # log_results_EB0(row,cols,save_name)
        #%%
        save_name = data_set_flags[flag_dataset]+'CNN'
        filename = os.path.join(sav_path,save_name+'.obj')
        y_test_pred_list = []
        rmse_list = []
        start_test = time.time()
        
        y_test_pred_list,rmse_list = model_serv(X_test_list,y_test_list,model,scaler,batch_size)
        # print(np.mean(rmse_list))
        
        end_test = time.time()
        test_time = end_test - start_test
        val_loss = history.history['val_loss']
        train_loss = history.history['loss']
        obj = {'test_time':test_time,'train_time':train_time
                ,'y_test':y_test_list,'y_test_pred':y_test_pred_list
                ,'scaler':scaler,'rmse_list':np.array(rmse_list)
                ,'cols':cols,'Mids_test':Mids_test,'val_loss':val_loss
                ,'train_loss':train_loss}
        save_object(obj, filename)
        
        print(row_alibaba)
