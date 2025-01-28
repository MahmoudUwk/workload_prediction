
def get_CEDL(encoder_inputs):
    from keras.layers import Dropout,Dense,Flatten, LSTM, AdditiveAttention, concatenate, RepeatVector

    num_layers = 1
    units = dense_units= 2**7
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
    outputs = Dense(dense_units, activation='swish')(decoder_combined_context)
    outputs = Dropout(0.3)(outputs)
    # outputs = Dense(output_dim, activation=None, name="output_layer")(outputs)

    # Flatten the output
    outputs = Flatten(name="flatten_output")(outputs)


    return outputs
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_parallel_timesnet_attention(input_shape, pred_len, top_k_frequencies=8, num_heads=2, lstm_units=32, dense_units=[32]):
    inputs = keras.Input(shape=input_shape)  # Input: (batch_size, sequence_length, num_features)

    # 1. Fast Fourier Transform (FFT) using a Lambda layer
    def fft_layer(x):
        x_cpu = x[:, :, 0]  # Extract CPU utilization
        x_fft = tf.signal.fft(tf.cast(x_cpu, tf.complex64))
        x_fft = tf.abs(x_fft)  # Get magnitude
        # Select top-k frequencies
        fft_vals, fft_ids = tf.math.top_k(x_fft, k=top_k_frequencies)
        return fft_vals  # (batch_size, top_k_frequencies)

    x_fft = layers.Lambda(fft_layer)(inputs)

    # 2. Dense Layers and Multi-Head Attention (for x_fft)
    # Apply Dense layer to x_fft
    x_fft_dense = layers.Dense(units=dense_units[0], activation="relu")(x_fft)  # (batch_size, dense_units[0])

    # Add a dummy sequence dimension using Lambda layer
    x_fft_processed = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x_fft_dense) # (batch_size, 1, dense_units[0])

    x_fft_processed = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x_fft_processed.shape[-1])(x_fft_processed, x_fft_processed)

    # Add dummy dimension to x_fft_dense using Lambda layer for residual connection
    x_fft_dense_expanded = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x_fft_dense) # (batch_size, 1, dense_units[0])

    x_fft_processed = layers.LayerNormalization(epsilon=1e-6)(x_fft_processed + x_fft_dense_expanded)  # Add & Norm (residual connection)

    # Remove the dummy sequence dimension using Lambda layer
    x_fft_processed = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(x_fft_processed)  # (batch_size, dense_units[0])

    # 3. LSTM Network (for x_time)
    x_time = inputs  # (batch_size, sequence_length, num_features)
    x_time_processed = get_CEDL(x_time)  # Process with LSTM

    # 4. Feature Fusion
    x = layers.concatenate([x_fft_processed, x_time_processed])

    # 5. Dense Layers
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # Output layer
    outputs = layers.Dense(pred_len)(x)  # Output: (batch_size, pred_len)

    # Reshape to (batch_size, 1, pred_len) using Lambda layer
    outputs = layers.Lambda(lambda x: tf.reshape(x, (-1, 1, pred_len)))(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


#%%
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import os
from keras.callbacks import EarlyStopping
import numpy as np
from utils_WLP import save_object,MAE,MAPE, expand_dims,get_en_de_lstm_model_attention,get_best_lsmt_para
from utils_WLP import get_dict_option,get_lstm_model,model_serv,RMSE,log_results_EB0
import time
scaler = 100
num_epoc = 7000
batch_size = 2**8

lr = 0.005
seq_len = 32

flag_datasets = [0]

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

    dense_units = [256]
    model = create_parallel_timesnet_attention(input_shape,1
                   ,dense_units=dense_units)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1)
                  ,metrics=[ 'mape'])     

    # from keras_lr_finder import LRFinder
    # lr_finder = LRFinder(model)
    # lr_finder.find(
    #     X_train,
    #     y_train,
    #     start_lr=0.001,  # Small starting learning rate
    #     end_lr=0.1,       # Large ending learning rate
    #     batch_size=batch_size,  # Batch size for training
    #     epochs=5 ,      # Number of epochs (typically small, 2-5 is often enough)
    #     # save_path='tmp.weights.h5'
    # )
    # optimal_lr = lr_finder.get_best_lr()
    # print(f"Suggested learning rate: {optimal_lr}")
    # model.compile(optimizer=Adam(learning_rate=optimal_lr), loss=Huber(delta=1)
    #               ,metrics=[ 'mape'])  


        # Callbacks
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=2e-4)

    callbacks_list = [EarlyStopping(monitor='val_loss', 
                        patience=40, restore_best_weights=True),reduce_lr]
    
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
    # rmse_i = RMSE(y_test,y_pred)
    # RMSE_list.append([(block,filter_i),rmse_i])
    # print(RMSE_list)
    # save_name = os.path.join(sav_path,'LSTMCNN.csv')
    # ['num_stagse','num_blocks','num_filters','kernel_size']
    # row = [rmse_i,stage,block,filter_i,kss_i]
    # log_results_EB0(row,cols,save_name)
    #%%
    # save_name = data_set_flags[flag_dataset]+'LSTMCNN'
    # filename = os.path.join(sav_path,save_name+'.obj')
    # y_test_pred_list = []
    # rmse_list = []
    # start_test = time.time()
    
    # y_test_pred_list,rmse_list = model_serv(X_test_list,y_test_list,model,scaler,batch_size)
    # # print(np.mean(rmse_list))
    
    # end_test = time.time()
    # test_time = end_test - start_test
    #val_loss = history.history['val_loss']
    # train_loss = history.history['loss']
    # obj = {'test_time':test_time,'train_time':train_time
    #         ,'y_test':y_test_list,'y_test_pred':y_test_pred_list
    #         ,'scaler':scaler,'rmse_list':np.array(rmse_list)
    #         ,'cols':cols,'Mids_test':Mids_test,'val_loss':val_loss
    #         ,'train_loss':train_loss}
    # save_object(obj, filename)
    print(row_alibaba)