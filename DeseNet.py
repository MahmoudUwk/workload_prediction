import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber


def dense_block(x, filters, growth_rate, dropout_rate=0.2):
    x_shortcut = x

    # Bottleneck layer (1x1 convolution)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.Dropout(dropout_rate)(x)

    # 3x1 convolution
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(growth_rate, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Concatenate with shortcut
    x = layers.Concatenate()([x_shortcut, x])

    return x

def transition_block(x, reduction=0.5, activation='relu'):
    """
    Transition Block as in DenseNet.

    Args:
        x: Input tensor.
        reduction: Compression factor to reduce the number of filters.
        activation: Activation function.

    Returns:
        Output tensor of the transition block.
    """
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv1D(int(x.shape[-1] * reduction), 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.AveragePooling1D(2, strides=2)(x)
    return x

def DenseNet_1D(input_shape, num_classes=1, blocks=[6, 12, 24, 16], growth_rate=32,
                compression=0.5, dropout_rate=0.2, activation='relu'):


    inputs = keras.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv1D(64, 7, strides=2, padding='same', use_bias=True, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    # Dense blocks and transition layers
    for i, num_blocks in enumerate(blocks):
        for _ in range(num_blocks):
            x = dense_block(x, 4 * growth_rate, growth_rate, dropout_rate)
        if i != len(blocks) - 1:
            x = transition_block(x, compression, activation)

    # Global average pooling and output layer
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='linear')(x)  # Linear activation for regression

    model = keras.Model(inputs, outputs)
    return model

# Example usage:
# input_shape = (32, 2)  # 32 time steps, 2 features (CPU and memory)
# model = DenseNet_1D(input_shape,blocks=[2,2,2])
# model.summary()
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
batch_size = 2**9
patientce_para = 20
lr = 0.009
seq_len =  32#get_best_lsmt_para(0)['seq']
#swish relu gelu
activation_func = 'swish'
flag_datasets = [0]

data_set_flags = ['Alibaba','google','BB']
filter_list = [0]
kss_sizes=[0]

#kss_i = kss_sizes[0]
blocks = [0]
num_stages = [0]
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
    
    # best_params =  {'units': 256,
    #  'num_layers': 2,
    #  'seq': 20,
    #  'dense_units': 256}
    
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
    for stage in num_stages:
        for block in blocks:
            for filter_i in filter_list:
                for kss_i in kss_sizes:
                


                    model = DenseNet_1D(input_shape,blocks=[1,1,1] 
                       , growth_rate=64,compression=0.5)
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
                    
                    
                    
                    
                    
                    rmse_i = RMSE(y_test,y_test_pred)
                    RMSE_list.append([(block,filter_i),rmse_i])
    
                    save_name = os.path.join(sav_path,'DensetNet.csv')
                    # ['num_stagse','num_blocks','num_filters','kernel_size']
                    row = [rmse_i,stage,block,filter_i,kss_i]
                    log_results_EB0(row,cols,save_name)
                    #%%
                    save_name = data_set_flags[flag_dataset]+'_effecientB0'
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
                            ,'best_params':row,'cols':cols,'Mids_test':Mids_test,'val_loss':val_loss
                            ,'train_loss':train_loss}
                    save_object(obj, filename)
                    
                    print(row_alibaba)
