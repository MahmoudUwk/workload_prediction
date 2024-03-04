# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:10:28 2024

@author: mahmo
"""
import keras
from keras import layers
import numpy as np

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    # x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    # x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1,activation='relu')(x)
    # outputs = layers.Dense(n_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)
#%%
from sklearn.preprocessing import MinMaxScaler
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(X)

X = scaler.transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)


# x_train, y_train = make_regression(n_samples=100, n_features=5, noise=1, random_state=42)

# x_test, y_test = make_regression(n_samples=100, n_features=5, noise=1, random_state=42)

x_train = np.expand_dims(x_train,axis=2)
x_test = np.expand_dims(x_test,axis=2)

input_shape = x_train.shape[1:]
#%%
model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.1,
    dropout=0.1,
)

model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=2e-3),
)
model.summary()

callbacks = []#[keras.callbacks.EarlyStopping(patience=120, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=900,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)
#%%
mse = mean_squared_error(y_test, model.predict(x_test))

print(mse)
#The mean squared error (MSE) on test set: 3044.4733
# print(model.evaluate(x_test, y_test, verbose=1))