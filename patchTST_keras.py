import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_CEDL(encoder_inputs,units):
    num_layers = 1
    # units = dense_units = 2**7

    # Encoder
    encoder_outputs = layers.LSTM(units, return_sequences=True)(encoder_inputs)
    for i in range(num_layers - 1):
        encoder_outputs = layers.LSTM(units, return_sequences=True, name=f"encoder_lstm_{i+1}")(encoder_outputs)
    encoder_outputs, state_h, state_c = layers.LSTM(units, return_sequences=True, return_state=True, name=f"encoder_lstm_{num_layers}")(encoder_outputs)
    encoder_states = [state_h, state_c]

    # Decoder
    context = layers.RepeatVector(1, name="context_vector")(encoder_outputs[:, -1, :])
    decoder = layers.LSTM(units, return_sequences=True, name="decoder_lstm_0")(context, initial_state=encoder_states)
    for i in range(num_layers - 1):
        decoder = layers.LSTM(units, return_sequences=True, name=f"decoder_lstm_{i+1}")(decoder)

    # Multi-Head Attention
    attention_layer = layers.AdditiveAttention(name="attention_layer")
    attention_output = attention_layer([decoder, encoder_outputs])
    
    # attention_layer = layers.MultiHeadAttention(num_heads=2, key_dim=units)
    # attention_output = attention_layer(query=decoder, key=encoder_outputs, value=encoder_outputs)  # Pass key and value

    # Concatenate and Dense Layers
    decoder_combined_context = layers.concatenate([decoder, attention_output], axis=-1)
    outputs = layers.Dense(units, activation='swish')(decoder_combined_context)
    # outputs = layers.Dropout(0.5)(outputs)
    outputs = layers.Flatten()(outputs)

    return outputs

def create_patch_tst_lstm_hybrid(input_shape, pred_len=1, 
                 patch_length=16, num_heads=4,  dense_units=256):

    inputs = keras.Input(shape=input_shape)  # Input shape: (sequence_length, num_features)
    sequence_length = input_shape[0]
    num_features = input_shape[1]

    # 1. Patch TST Block
    # Divide the time series into patches
    num_patches = sequence_length // patch_length
    x_patches = layers.Reshape((num_patches, patch_length * num_features))(inputs)  # Shape: (batch_size, num_patches, patch_length * num_features)
    
    # Transformer Encoder for Patch TST
    transformer_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=patch_length * num_features)(x_patches, x_patches)
    transformer_output = layers.LayerNormalization(epsilon=1e-6)(transformer_output + x_patches)  # Add & Norm
    transformer_output = layers.GlobalAveragePooling1D()(transformer_output)  # Aggregate patch-level features

    # 2. LSTM Block
    x_lstm = get_CEDL(inputs,dense_units)
    x = layers.concatenate([transformer_output, x_lstm])  # Combine Patch TST and LSTM outputs

    # 4. Dense Layers

    x = layers.Dense(dense_units, activation="swish")(x)
    outputs = layers.Dense(pred_len)(x)  # Predict the next `pred_len` time steps
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


#%%
from keras.optimizers import Adam
import os
from keras.callbacks import EarlyStopping
import numpy as np
from utils_WLP import save_object,MAE,MAPE, expand_dims,get_en_de_lstm_model_attention,get_best_lsmt_para
from utils_WLP import get_dict_option,get_lstm_model,model_serv,RMSE,log_results_EB0
import time
scaler = 100
num_epoc = 7000
batch_size = 2**9
dense_units = [64,128,256,512]
num_heads = [4,6,8]
lr = 0.001
seq_len = 32

flag_datasets = [2]

data_set_flags = ['Alibaba','google','BB']

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

    cols = ['RMSE','MAE','MAPE','Units','numhead']
    save_name = os.path.join(sav_path,'hybrid.csv')
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

    for units_den_i in dense_units:
        for num_head in num_heads:
            model = create_patch_tst_lstm_hybrid(input_shape,num_heads=num_head
                           ,dense_units=units_den_i)
            model.summary()
            model.compile(optimizer=Adam(learning_rate=lr), loss='MAE'
                          ,metrics=[ 'mape'])          
                # Callbacks
            from tensorflow.keras.callbacks import ReduceLROnPlateau
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=2e-4)
        
            callbacks_list = [EarlyStopping(monitor='val_loss', 
                                patience=15, restore_best_weights=True),reduce_lr]
            
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
                           ,MAPE(y_test*scaler,y_test_pred),units_den_i,num_head]

            flag = log_results_EB0(row_alibaba,cols,save_name)
            #%%
            if flag == 1 :
                save_name = data_set_flags[flag_dataset]+'TST_LSTM'
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