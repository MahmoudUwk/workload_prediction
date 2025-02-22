import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers#, regularizers
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber

def conv_block(x, filters, kernel_size, strides=1, activation='relu', use_bn=True):
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=not use_bn)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def simplified_inverted_residual_block(x, filters, kernel_size=3, strides=1, activation='silu', use_bn=True, se_ratio=None, survival_probability=None):
    # Store input ten sor for residual connection
    inputs = x
    
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

    # Stochastic Depth
    if survival_probability is not None:
        x_project = layers.Dropout(1 - survival_probability)(x_project)

    # Residual connection (only if strides are 1 and input/output channels are the same)
    if strides == 1 and inputs.shape[-1] == filters:
        x = layers.Add()([inputs, x_project])  # Changed to Add instead of Concatenate
    else:
        x = x_project
    return x

def squeeze_excite_block(x, se_ratio=0.25):
    channels = x.shape[-1]
    se_shape = (1, channels)

    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(int(channels * se_ratio), activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    x = layers.Multiply()([x, se])
    return x

def Improved_EfficientNet_Like_1D(input_shape
          , num_classes=1, base_filters=128, blocks_per_stage=[1, 2, 2],
          kss=[3, 3, 3], activation='silu', use_se=True, use_bn=True):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Initial convolution
    x = conv_block(x, base_filters, kernel_size=3, strides=1, activation=activation, use_bn=use_bn)

    # Calculate dropout and survival probabilities
    num_blocks = sum(blocks_per_stage)
    drop_rate = 0.2  # Initial dropout rate
    survival_probs = [1 - drop_rate * float(i) / num_blocks for i in range(num_blocks)]
    block_idx = 0

    # Inverted residual blocks
    for i, num_blocks in enumerate(blocks_per_stage):
        num_filters = base_filters * (2 ** i)
        for j in range(num_blocks):
            # dropout_rate = drop_rate * float(block_idx) / num_blocks  # Progressive dropout
            x = simplified_inverted_residual_block(
                x, num_filters, 
                kernel_size=kss[i],
                activation=activation,
                use_bn=use_bn,
                se_ratio=0.5 if use_se else None,
                survival_probability=survival_probs[block_idx]
            )
            block_idx += 1

    # Global Average Pooling with attention
    attention = layers.Dense(1, activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    x = layers.GlobalAveragePooling1D()(x)

    # Dropout with progressive rate
    # x = layers.Dropout(0.5)(x)

    # Dense layers with regularization
    x = layers.Dense(
        128, 
        activation=activation,
        # kernel_regularizer=keras.regularizers.l2(0.01),
        kernel_initializer='he_normal'
    )(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation=None,
        # kernel_regularizer=keras.regularizers.l2(0.01)
    )(x)

    model = keras.Model(inputs, outputs)
    return model

#%%
import numpy as np
import os
from keras.callbacks import EarlyStopping
from utils_WLP import save_object,MAE,MAPE, expand_dims,get_en_de_lstm_model_attention,get_best_lsmt_para
from utils_WLP import get_dict_option,get_lstm_model,model_serv,RMSE,log_results_EB0
import time
scaler = 100
num_epoc = 7000
batch_size = 2**14
use_bn = True
lr = 0.001
seq_len = 32# get_best_lsmt_para(0)['seq']
#relu relu gelu
activation_func = 'swish'
flag_datasets = [2]

data_set_flags = ['Alibaba','google','BB']
filter_list = [2**6]
kss_sizes=[3]

#kss_i = kss_sizes[0]
blocks = [3]
num_stages = [3]
cols = ['RMSE','MAE','MAPE','num_stagse','num_blocks','num_filters','kernel_size']

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
    for stage in num_stages:
        for block in blocks:
            for filter_i in filter_list:
                for kss_i in kss_sizes:
                
                    blocks_per_stage= [1]*(stage-1)+[block]
                    kss = [max(kss_i-2*i,3) for i in range(stage)]
                    
                    model = Improved_EfficientNet_Like_1D(
                              input_shape=input_shape,use_bn=use_bn , num_classes=1,kss=kss 
                              ,base_filters=filter_i, activation=activation_func
                              , blocks_per_stage =blocks_per_stage)
                    # model = Improved_EfficientNet_Like_1D(input_shape)
                    model.summary()
                    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1)
                                  ,metrics=[ 'mape'])                
                    callbacks_list = [EarlyStopping(monitor='val_loss', 
                                        patience=30, restore_best_weights=True)]
                    
                    start_train = time.time()
                    history = model.fit(X_train, y_train, epochs=num_epoc , 
                              batch_size=batch_size, verbose=2, shuffle=True, 
                              validation_data=(X_val,y_val),callbacks=callbacks_list)
                    
                    end_train = time.time()
                    train_time = (end_train - start_train)/60
                    
                    X_test,y_test
                    # X_test,y_test
                    y_test_pred = (model.predict(X_test))*scaler
                    row_alibaba = [RMSE(y_test*scaler,y_test_pred),MAE(y_test*scaler,y_test_pred)
                                   ,MAPE(y_test*scaler,y_test_pred)]

                    save_name = os.path.join(sav_path,'EfficientB0.csv')
                    # ['num_stagse','num_blocks','num_filters','kernel_size']
                    row = row_alibaba+[stage,block,filter_i,kss_i]
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
                            ,'cols':cols,'Mids_test':Mids_test,'val_loss':val_loss
                            ,'train_loss':train_loss}
                    save_object(obj, filename)
                    print(row_alibaba)



