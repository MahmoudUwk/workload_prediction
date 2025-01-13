import os
import numpy as np
import pandas as pd
from args import get_paths
import sklearn
from tsai.all import *
import torch
import pickle
from tsai.callback.core import SaveModelCallback
import time
from tsai.callback.core import EarlyStoppingCallback

# Function to calculate RMSE
def get_InceptionTimePlus(trial):

    arch = dict(
         nf = trial.suggest_categorical("nf", [32,64,128,256,512]),
         ks= trial.suggest_categorical("ks", [3,5,7,11,17,19,31]),  # Example: Suggest different kernel size combinations
         bn_1st= True,
         sa = True,
    )
    return arch
def get_my_RMSE(test, pred):
    return np.sqrt(np.mean((np.squeeze(np.array(test)) - np.squeeze(np.array(pred))) ** 2))

# Data Preprocessing
def expand_dims(X):
    return np.expand_dims(X, axis=len(X.shape))

from utils_WLP import save_object, expand_dims,get_en_de_lstm_model_attention
from utils_WLP import get_dict_option,get_lstm_model,model_serv
flag_dataset = 0
data_set_flags = ['Alibaba','google','BB']

if data_set_flags[flag_dataset] == 'Alibaba':
    from args import get_paths
elif data_set_flags[flag_dataset]=='google':
    from args_google import get_paths
elif data_set_flags[flag_dataset]=='BB':
    from args_BB import get_paths

base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()

batch_size=2**7
train_dict , val_dict, test_dict = get_dict_option(data_set_flags[flag_dataset],20)
X_train = train_dict['X'][:,:,0]
y_train = train_dict['Y']
X_val = val_dict['X'][:,:,0]
y_val = val_dict['Y']
X_test = test_dict['X'][:,:,0]
y_test = test_dict['Y']
prediction_horizon = 1  # How many steps ahead to predict
scaler=100
X_train = np.expand_dims(np.squeeze(X_train), axis=1)
y_train = expand_dims(expand_dims(y_train))
y_val = expand_dims(expand_dims(y_val))
X_val = np.expand_dims(np.squeeze(X_val), axis=1)
X_test = np.expand_dims(np.squeeze(X_test), axis=1)

X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])
#%%
# model_names = ['ResNetPlus','FCNPlus','InceptionTimePlus']
from arch_config_all import get_ResNetPlus,get_FCNPlus,get_InceptionTimePlus 

model_name = 'InceptionTimePlus'
# Create the best model

    
def objective(trial):
    cbs = [
        EarlyStoppingCallback(monitor='valid_loss', min_delta=1e-6, patience=15),
        SaveModelCallback(monitor='valid_loss', fname=model_name, every_epoch=False)  # Save the best model
    ]

    arch_config = get_InceptionTimePlus(trial)
    learn = TSForecaster(
        X,
        y,
        splits=splits,
        batch_size=batch_size,  # You used 128 in the original evaluation code
        path=os.path.join(sav_path, model_name),
        arch=model_name,
        arch_config=arch_config,
        metrics=[rmse],
        cbs=cbs,
    )
    lr_to_use = learn.lr_find().valley


    learn.fit_one_cycle(2500, lr_to_use)###########################
    # Get the best validation loss from the recorder
    val_rmse_all = np.array(learn.recorder.values)[:,-1]
    best_epoch_index = val_rmse_all.argmin(axis=0) # Index of the best epoch
    best_val_loss = learn.recorder.values[best_epoch_index][-1]  # Best valid loss
    best_epoch = best_epoch_index + 1 # + 1 because epoch count starts from 1

    # Save the model
    best_weights_path = os.path.join(trial_dir, "learn.pth")
    learn.save(best_weights_path)

    # Save trial metadata
    with open(os.path.join(trial_dir, "metadata.txt"), "w") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best valid loss: {best_val_loss}\n")
        f.write(f"Architecture config: {str(arch_config)}\n")

    # Store trial results in user attributes for later retrieval (Corrected)
    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("best_weights_path", best_weights_path)
    # Clean up
    del learn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Report the best validation loss to Optuna
    return best_val_loss




