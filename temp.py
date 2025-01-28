import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class FFTOperations(layers.Layer):
    def __init__(self, top_k_frequencies=8, window_type='hann', **kwargs):
        super(FFTOperations, self).__init__(**kwargs)
        self.top_k_frequencies = top_k_frequencies
        self.window_type = window_type

    def call(self, inputs):
        x_cpu = inputs[:, :, 0]  # Shape: (batch_size, sequence_length)

        # Apply window function
        if self.window_type == 'hann':
            window = tf.signal.hann_window(tf.shape(x_cpu)[1], periodic=True)
        elif self.window_type == 'hamming':
            window = tf.signal.hamming_window(tf.shape(x_cpu)[1], periodic=True)
        else:
            window = tf.ones_like(x_cpu[0])  # No window

        x_cpu_windowed = x_cpu * window

        # Perform FFT
        x_fft = tf.signal.fft(tf.cast(x_cpu_windowed, tf.complex64))
        x_fft_magnitude = tf.abs(x_fft)
        x_fft_phase = tf.math.angle(x_fft)

        # Select top-k frequencies
        fft_vals, fft_ids = tf.math.top_k(x_fft_magnitude, k=self.top_k_frequencies)
        # Reconstruct the time-domain signal using Inverse FFT
        batch_size = tf.shape(x_cpu)[0]
        seq_len = tf.shape(x_cpu)[1]

        # Initialize a zero tensor for the full FFT spectrum
        x_padded = tf.zeros([batch_size, seq_len], dtype=tf.complex64)

        # Scatter the top-k frequencies into the zero tensor
        indices = tf.stack([tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, self.top_k_frequencies]), fft_ids], axis=-1)
        updates = tf.cast(fft_vals, tf.complex64)
        x_padded = tf.tensor_scatter_nd_update(x_padded, indices, updates)

        # Perform Inverse FFT
        x_reconstructed = tf.signal.ifft(x_padded)  # Shape: (batch_size, sequence_length)
        x_reconstructed = tf.math.real(x_reconstructed)  # Return real part

        # Add a feature dimension to the reconstructed signal
        x_reconstructed = tf.expand_dims(x_reconstructed, axis=-1)  # Shape: (batch_size, sequence_length, 1)

        return x_reconstructed

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)  # Output shape: (batch_size, sequence_length, 1)

# LSTM-based Encoder-Decoder with Attention
def get_CEDL(encoder_inputs):
    num_layers = 1
    units = 2**6
    dense_units = 2**6

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
    # attention_layer = layers.AdditiveAttention(name="attention_layer")
    # attention_output = attention_layer([decoder, encoder_outputs])
    
    attention_layer = layers.MultiHeadAttention(num_heads=2, key_dim=units)
    attention_output = attention_layer(query=decoder, key=encoder_outputs, value=encoder_outputs)  # Pass key and value

    # Concatenate and Dense Layers
    decoder_combined_context = layers.concatenate([decoder, attention_output], axis=-1)
    outputs = layers.Dense(dense_units, activation='gelu')(decoder_combined_context)
    outputs = layers.Dropout(0.25)(outputs)
    outputs = layers.Flatten()(outputs)

    return outputs


# Main Model
def create_parallel_timesnet_attention(input_shape, pred_len, top_k_frequencies=3, 
                   num_heads=2, lstm_units=32, dense_units=[32]):
    inputs = keras.Input(shape=input_shape)  # Input: (batch_size, sequence_length, num_features)

    # 1. FFT/IFFT Operations
    fft_layer = FFTOperations(top_k_frequencies=top_k_frequencies, window_type='hann')
    x_reconstructed = fft_layer(inputs)  # Shape: (batch_size, sequence_length, 1)

    # 2. Dense Layers and Multi-Head Attention (for x_fft)
    x_fft_dense = layers.Dense(units=dense_units[0], activation="swish")(x_reconstructed[:, -1, :])  # Use the last time step
    x_fft_processed = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x_fft_dense)  # Shape: (batch_size, 1, dense_units[0])
    
    x_fft_processed = layers.MultiHeadAttention(num_heads=num_heads, 
                key_dim=x_fft_processed.shape[-1])(x_fft_processed, x_fft_processed)
    x_fft_dense_expanded = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x_fft_dense)  # Shape: (batch_size, 1, dense_units[0])
    x_fft_processed = layers.LayerNormalization(epsilon=1e-6)(x_fft_processed + x_fft_dense_expanded)  # Add & Norm
    x_fft_processed = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(x_fft_processed)  # Shape: (batch_size, dense_units[0])

    # 3. LSTM Network (for x_time)
    x_time = inputs  # Shape: (batch_size, sequence_length, num_features)
    x_time_processed = get_CEDL(x_time)  # Process with LSTM

    # 4. Feature Fusion
    x_fft_weighted = layers.Dense(1)(x_fft_processed)
    x_time_weighted = layers.Dense(1)(x_time_processed)
    x_recon_weighted = layers.Dense(1)(x_reconstructed[:, -1, :])
    x = layers.Add()([x_fft_weighted, x_time_weighted, x_recon_weighted])
    # x = layers.concatenate([x_fft_processed, x_time_processed, x_reconstructed[:, -1, :]])  # Use the last time step of the reconstructed signal

    # 5. Dense Layers
    # for units in dense_units:
    #     x = layers.Dense(units, activation="swish")(x)
    # x = layers.Dropout(0.25)(x)

    # Output layer
    outputs = layers.Dense(pred_len)(x)  # Output: (batch_size, pred_len)
    outputs = layers.Lambda(lambda x: tf.reshape(x, (-1, 1, pred_len)))(outputs)  # Reshape to (batch_size, 1, pred_len)

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
batch_size = 2**7

lr = 0.001
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

    dense_units = [128]
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