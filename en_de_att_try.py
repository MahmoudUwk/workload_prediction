# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:37:37 2024

@author: Admin
"""
import random
import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    multiply, concatenate, Flatten, Activation, dot
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
# import pydot as pyd
from keras.utils.vis_utils import plot_model, model_to_dot
# keras.utils.vis_utils.pydot = pyd
from Alibaba_fet_features_LSTM import get_dataset_alibaba_lstm
#%%
def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))
def RMSE(test,pred):
    return np.sqrt(np.mean((np.squeeze(test) - np.squeeze(pred))**2))
#%%
seq_length = 3
cluster_num = 'all'
X_train,y_train,X_val,y_val,X_test ,y_test,scaler,clusters = get_dataset_alibaba_lstm(seq_length,cluster_num)
ind_rand = np.random.permutation(len(X_train))
X_train = X_train[ind_rand]
y_train = y_train[ind_rand]
y_test = y_test*scaler
input_dim=(X_train.shape[1],X_train.shape[2])
print(X_train.shape)
output_dim = 1
print(input_dim,output_dim)
y_train = expand_dims(expand_dims(y_train))
y_val = expand_dims(expand_dims(y_val))
if len(X_train.shape)<3:
    X_train = expand_dims(X_train)
    X_val = expand_dims(X_val)
    X_test = expand_dims(X_test)
print(X_train.shape,y_train.shape,X_test.shape,X_val.shape)
#%%
n_hidden = 512
input_train = Input(shape=(X_train.shape[1], X_train.shape[2]))
output_train = Input(shape=(1))
encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
    n_hidden,
    return_state=True, return_sequences=True)(input_train)
encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)
decoder_input = RepeatVector(X_train.shape[1])(encoder_last_h)
decoder_stack_h = LSTM(n_hidden,
 return_state=False, return_sequences=True)(
 decoder_input, initial_state=[encoder_last_h, encoder_last_c])
     
 
attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
attention = Activation('softmax')(attention)
context = dot([attention, encoder_stack_h], axes=[2,1])
context = BatchNormalization(momentum=0.6)(context)

decoder_combined_context = concatenate([context, decoder_stack_h])

out = TimeDistributed(Dense(12))(decoder_combined_context)
out = Flatten()(out)
out = Dense(output_dim)(out)


model = Model(inputs=input_train, outputs=out)
# opt = Adam(lr=0.001, clipnorm=1)
model.compile(loss='mse', optimizer='adam')
model.summary()

epochs_num = 150
batch_size_n = 2**10
val_split_size = 0
callbacks_list = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
if X_val.size ==0:
    history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_split=val_split_size,callbacks=callbacks_list)
else:
    print("val provided")
    history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=2, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
              
#%%
y_test_pred = model.predict(X_test)*scaler



rmse = RMSE(y_test,y_test_pred)
print(rmse)