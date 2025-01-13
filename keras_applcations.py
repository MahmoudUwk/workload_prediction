import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam

def conv_block(x, filters, kernel_size, strides=1, activation='swish', use_bn=True):
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=not use_bn)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def simplified_inverted_residual_block(x, filters, kernel_size=3, strides=1, activation='swish', use_bn=True, se_ratio=None):
    # Depthwise convolution
    x_dw = layers.DepthwiseConv1D(kernel_size, strides=strides, padding='same', use_bias=not use_bn)(x)
    if use_bn:
        x_dw = layers.BatchNormalization()(x_dw)
    x_dw = layers.Activation(activation)(x_dw)

    # Squeeze and excite (if enabled)
    if se_ratio is not None:
        x_dw = squeeze_excite_block(x_dw, se_ratio)

    # Projection phase (1x1 convolution)
    x_project = layers.Conv1D(filters, 1, padding='same', use_bias=not use_bn)(x_dw)
    if use_bn:
        x_project = layers.BatchNormalization()(x_project)

    # Residual connection (only if strides are 1 and input/output channels are the same)
    if strides == 1 and x.shape[-1] == filters:
        x = layers.Add()([x, x_project])
    else:
        x = x_project
    return x

def squeeze_excite_block(x, se_ratio=0.25):
    channels = x.shape[-1]
    se_shape = (1, channels)

    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(int(channels * se_ratio), activation='swish')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    x = layers.Multiply()([x, se])
    return x
#swish relu
def Simplified_EfficientNet_Like_1D(input_shape, num_classes=1, base_filters=16, blocks_per_stage=[1, 1, 1],
                                    kss=[3, 3, 3], activation='swish', use_se=True):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Initial convolution
    x = conv_block(x, base_filters, kernel_size=3, strides=1, activation=activation)  # Increased stride to 2

    # Simplified Inverted Residual Blocks
    for i, num_blocks in enumerate(blocks_per_stage):
        num_filters = base_filters * (2 ** i)
        for j in range(num_blocks):
            x = simplified_inverted_residual_block(x, num_filters, kernel_size=kss[i],
                                                    strides=2 if j == 0 and i < 2 else 1,  # Added strides here
                                                    activation=activation,
                                                    se_ratio=0.25 if use_se else None)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='linear')(x)

    model = keras.Model(inputs, outputs)
    return model

#%%

import os
from keras.callbacks import EarlyStopping
import numpy as np
from utils_WLP import save_object, expand_dims,get_en_de_lstm_model_attention
from utils_WLP import get_dict_option,get_lstm_model,model_serv,RMSE,log_results_EB0
import time
scaler = 100
num_epoc = 7000
batch_size = 2**8
lr = 0.001
    


flag_datasets = [0]

data_set_flags = ['Alibaba','google','BB']
filter_list = [2**10]
kss_sizes=[3,5]

kss_i = kss_sizes[0]
blocks = [1]
num_stages = [3]
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
    
    train_dict , val_dict, test_dict = get_dict_option(data_set_flags[flag_dataset],20)
    
    
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
                
            # input_shape, num_classes=1, base_filters=16, blocks_per_stage=[1, 1, 1],
            #    kss=[3, 3, 3], activation='relu', use_se=True

                    blocks_per_stage= [block]+[1]*(stage-1)
                    model = Simplified_EfficientNet_Like_1D(input_shape, num_classes=1
                                    ,kss=[kss_i, kss_i, kss_i] ,base_filters=filter_i,
                                    blocks_per_stage =blocks_per_stage)
                    model.summary()
                    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
                
                    callbacks_list = [EarlyStopping(monitor='val_loss', 
                                        patience=15, restore_best_weights=True)]
                    
                    start_train = time.time()
                    history = model.fit(X_train, y_train, epochs=num_epoc , 
                              batch_size=batch_size, verbose=2, shuffle=True, 
                              validation_data=(X_val,y_val),callbacks=callbacks_list)
                    
                    end_train = time.time()
                    train_time = (end_train - start_train)/60
                    
                    X_test,y_test
                    y_pred = model.predict(X_test)
                    rmse_i = RMSE(y_test,y_pred)
                    RMSE_list.append([(block,filter_i),rmse_i])
                    print(RMSE_list)
                    save_name = os.path.join(sav_path,'EfficientB0.csv')
                    # ['num_stagse','num_blocks','num_filters','kernel_size']
                    row = [rmse_i,stage,block,filter_i,kss_i]
                    log_results_EB0(row,cols,save_name)
                    #%%
                    # save_name = data_set_flags[flag_dataset]+'_effecientB0'
                    # filename = os.path.join(sav_path,save_name+'.obj')
                    # y_test_pred_list = []
                    # rmse_list = []
                    # start_test = time.time()
                    
                    # y_test_pred_list,rmse_list = model_serv(X_test_list,y_test_list,model,scaler,batch_size)
                    # print(np.mean(rmse_list))
                    
                    # end_test = time.time()
                    # test_time = end_test - start_test
                    # val_loss = history.history['val_loss']
                    # train_loss = history.history['loss']
                    # obj = {'test_time':test_time,'train_time':train_time
                    #         ,'y_test':y_test_list,'y_test_pred':y_test_pred_list
                    #         ,'scaler':scaler,'rmse_list':np.array(rmse_list)
                    #         ,'best_params':row,'cols':cols,'Mids_test':Mids_test,'val_loss':val_loss
                    #         ,'train_loss':train_loss}
                    # save_object(obj, filename)
