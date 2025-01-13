import os
import numpy as np
import pandas as pd
from Alibaba_helper_functions import loadDatasetObj, list_to_array,flatten,save_object
from Alibaba_fet_features_LSTM_no_cluster import get_dataset_alibaba_lstm_no_cluster
from args import get_paths
import sklearn
from tsai.basics import *
import optuna
import gc
import torch
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
import sqlite3
from optuna.samplers import TPESampler
import pickle
from tsai.callback.core import SaveModelCallback
import time
from tsai.callback.core import EarlyStoppingCallback
def get_google_pred(seq_len,best_model,sav_path,batch_size):
    from Alibaba_helper_functions import loadDatasetObj, list_to_array,flatten
    from get_data_infere_py import get_data_inf
    scaler = 100
    feat_names = ['cpu_utilization', 'memory_utilization']
    target = ['cpu_utilization']
    X_list,Y_list,M_ids = get_data_inf(seq_len,feat_names,target)
    y_test_pred_list = []
    rmse_list_google = []
    for c,test_sample_all in enumerate(X_list):
        print(c)

        if len(test_sample_all)>batch_size:
            test_sample_all = split_3d_array(test_sample_all, batch_size)
        else:
            test_sample_all = [test_sample_all]
        pred_i = []
        # if c == 46:
        #     dasfas
        for test_sample in test_sample_all:

            test_sample = test_sample[:,:,0]

            test_sample = np.expand_dims(test_sample, axis=1)
            pred_ii, *_ = best_model.get_X_preds(test_sample/scaler)
            pred_ii = list(np.squeeze(to_np(pred_ii)*scaler)) if len(pred_ii) !=1 else [np.squeeze(to_np(pred_ii)*scaler)]
            pred_i.append(pred_ii)
    
        pred_i = flatten(pred_i)
    
        y_test_pred_list.append(pred_i)
        rmse_i_list = get_my_RMSE(Y_list[c],pred_i)
    
        rmse_list_google.append(rmse_i_list)
    
    obj = {'y_test':Y_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list_google),'Mids_test':M_ids}
    filename = os.path.join(sav_path,'TST_google.obj')
    save_object(obj, filename)
    # [len(i) for i in  y_test_pred_list]
    print(np.mean(rmse_list_google))
#%% bitbrain

def get_BB_pred(seq_len,best_model,sav_path,batch_size):
    from Alibaba_helper_functions import loadDatasetObj, list_to_array,flatten
    from get_data_infere_py_BB import get_data_inf_BB
    feat_names = ['CPU usage [%]', 'memory_utilization']
    target = 'CPU usage [%]'
    id_m = "machine_id"
    sort_by = 'Timestamp [ms]'
    X_list,Y_list,M_ids = get_data_inf_BB(seq_len,feat_names,target)
    y_test_pred_list = []
    rmse_list_BB = []
    for c,test_sample_all in enumerate(X_list):
        if len(test_sample_all)>batch_size:
            test_sample_all = split_3d_array(test_sample_all, batch_size)
        else:
            test_sample_all = [test_sample_all]
        pred_i = []
        for test_sample in test_sample_all:
            test_sample = test_sample[:,:,0]
            
            test_sample = np.expand_dims(test_sample, axis=1)
            pred_ii, *_ = best_model.get_X_preds(test_sample/scaler)
            pred_ii = list(np.squeeze(to_np(pred_ii)*scaler)) if len(pred_ii) !=1 else [np.squeeze(to_np(pred_ii)*scaler)]
            pred_i.append(pred_ii)
        pred_i = flatten(pred_i)
        #print(len(pred_i))
        y_test_pred_list.append(pred_i)
        rmse_i_list = get_my_RMSE(Y_list[c],pred_i)
        rmse_list_BB.append(rmse_i_list)
    
    obj = {'y_test':Y_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list_BB),'Mids_test':M_ids}
    filename = os.path.join(sav_path,'TST_BB.obj')
    save_object(obj, filename)
    
    print(np.mean(rmse_list_BB))

# Function to calculate RMSE
def get_my_RMSE(test, pred):
    return np.sqrt(np.mean((np.squeeze(np.array(test)) - np.squeeze(np.array(pred))) ** 2))

# Data Preprocessing
def expand_dims(X):
    return np.expand_dims(X, axis=len(X.shape))
def split_3d_array(array_3d, batch_size):
    num_samples = array_3d.shape[0]
    sub_arrays = []
    for i in range(0, num_samples, batch_size):
        sub_array = array_3d[i:i + batch_size]
        sub_arrays.append(sub_array)
    return sub_arrays

# Get paths (replace get_paths with your actual implementation)
_, _, _, _, _, sav_path, _ = get_paths()

# Constants for Optuna study
STUDY_NAME = "TST_study"
DB_PATH = os.path.join(sav_path, "TST_study.db")
MODELS_DIR = os.path.join(sav_path, "TST_study", "models")

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Hyperparameters for the data
seq_len = 20  # Sequence length
prediction_horizon = 1  # How many steps ahead to predict


# Load (or process and cache) the dataset
X_train, y_train, X_val, y_val, X_test_list, y_test_list, scaler, Mids_test = get_dataset_alibaba_lstm_no_cluster(seq_len, prediction_horizon)



X_train = np.expand_dims(np.squeeze(X_train), axis=1)
y_train = expand_dims(expand_dims(y_train))

y_val = expand_dims(expand_dims(y_val))
X_val = np.expand_dims(np.squeeze(X_val), axis=1)

X_test, y_test = list_to_array(X_test_list, seq_len, 1), list_to_array(
    y_test_list, 0, 1
)
X_test = np.expand_dims(np.squeeze(X_test), axis=1)

# Combine training and validation data for splitting in TSForecaster
X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])


# Optuna objective function
def objective(trial):
    # Fixed parameters
    dropout = 0.2
    padding_patch = True
    batch_size = 2**9
    n_epochs = 500  # Maximum number of epochs (early stopping will likely stop sooner)

    # Hyperparameters to be optimized by Optuna
    n_heads = trial.suggest_int("n_heads", 4, 8)
    d_model_multiplier = 2**trial.suggest_int("d_model_multiplier", 6, 7)
    d_model = n_heads * d_model_multiplier 

    arch_config = dict(
        n_layers=trial.suggest_int("n_layers", 2, 4),
        n_heads=n_heads,
        d_model=d_model,
        d_ff= 2**trial.suggest_int("d_ff", 8, 10, log=True),
        attn_dropout=0,
        dropout=dropout,
        patch_len=trial.suggest_int("patch_len", 2, 26),
        stride=trial.suggest_int("stride", 1, 8),
        padding_patch=padding_patch,
    )

    # Use mixed precision and early stopping callbacks
    cbs = [
        EarlyStoppingCallback(monitor='valid_loss', min_delta=1e-6, patience=10),
        SaveModelCallback(monitor='valid_loss', fname='best_model', every_epoch=False)  # Save the best model
    ]

    # Create the TSForecaster model
    learn = TSForecaster(
        X,
        y,
        splits=splits,
        batch_size=batch_size,
        path=os.path.join(sav_path, "model_tst"),
        arch="PatchTST",
        arch_config=arch_config,
        metrics=[rmse],
        cbs=cbs,
    )


    lr_to_use =  learn.lr_find().valley

    # Create a directory for the current trial
    trial_dir = os.path.join(MODELS_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # Train the model with fit_one_cycle (or fit)
    # The EarlyStoppingCallback will stop training when validation loss stops improving
    learn.fit_one_cycle(n_epochs, lr_to_use)

    # Get the best validation loss from the recorder
    val_rmse_all = np.array(learn.recorder.values)[:,-1]
    best_epoch_index = val_rmse_all.argmin(axis=0) # Index of the best epoch
    best_val_loss = learn.recorder.values[best_epoch_index][-1]  # Best valid loss
    best_epoch = best_epoch_index + 1 # + 1 because epoch count starts from 1

    # Save the model
    best_weights_path = os.path.join(trial_dir, "best_model.pth")
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

# Function to evaluate the best model found by Optuna
def evaluate_best_model(study,sav_path,batch_size):
    

    trial = study.best_trial
    


    # Reconstruct the architecture configuration from the best trial's parameters
    best_n_heads = trial.params["n_heads"] 
    best_d_model = best_n_heads *  2**trial.params["d_model_multiplier"]
    best_arch_config = dict(
        n_layers= trial.params["n_layers"], #4,#
        n_heads=best_n_heads,
        d_model=best_d_model,
        d_ff= trial.params["d_ff"], #512,#
        attn_dropout=0,
        dropout= 0.2,# You had a fixed dropout in the original code
        patch_len= trial.params["patch_len"], #12,#
        stride= trial.params["stride"],
        padding_patch=True,
    )
    cbs = [
        EarlyStoppingCallback(monitor='valid_loss', min_delta=1e-6, patience=15),
        SaveModelCallback(monitor='valid_loss', fname='best_model', every_epoch=False)  # Save the best model
    ]
    
    # Create the best model
    best_model = TSForecaster(
        X,
        y,
        splits=splits,
        batch_size=batch_size,  # You used 128 in the original evaluation code
        path=os.path.join(sav_path, "model_tst"),
        arch="PatchTST",
        arch_config=best_arch_config,
        metrics=[rmse],
        cbs=cbs,
    )
    lr_to_use = 0.008# best_model.lr_find().valley
    # Load the best weights
    start_test = time.time()
    
    best_model.fit_one_cycle(2500, lr_to_use)###########################
    
    end_test = time.time()
    train_time = end_test - start_test

    y_test_pred_list = []
    rmse_list = []
    start_test = time.time()
    scaler = 100
    for c,test_sample in enumerate(X_test_list):
        test_sample = np.expand_dims(np.squeeze(test_sample), axis=1)
        pred_i, *_ = best_model.get_X_preds(test_sample)
        pred_i = to_np(pred_i)        
    
        y_test_pred_list.append(pred_i*scaler)
        rmse_i_list = get_my_RMSE(y_test_list[c]*scaler,pred_i*scaler)
        y_test_list[c] = y_test_list[c]*scaler
        rmse_list.append(rmse_i_list)
    end_test = time.time()
    test_time = end_test - start_test
    filename = os.path.join(sav_path,'TST_ali.obj')
    obj = {'y_test':y_test_list,'y_test_pred':y_test_pred_list,
           'scaler':scaler,'rmse_list':np.array(rmse_list),
           'Mids_test':Mids_test,'train_time':train_time/60,'test_time':test_time}
    save_object(obj, filename)
    
    print(np.mean(rmse_list))
    #get_google_pred(seq_len,best_model,sav_path,batch_size)
    #get_BB_pred(seq_len,best_model,sav_path,batch_size)


# Function to get the training history of a specific trial
def get_trial_history(study, trial_number):
    trial = study.trials[trial_number]
    trial_dir = os.path.join(MODELS_DIR, f"trial_{trial_number}")
    
    if os.path.exists(trial_dir):
        with open(os.path.join(trial_dir, "metadata.txt"), "r") as f:
            metadata = f.read()
        
        return {
            "trial_number": trial_number,
            "params": trial.params,
            "value": trial.value,
            "best_epoch": trial.user_attrs.get("best_epoch"),
            "epoch_losses": trial.user_attrs.get("epoch_losses"),
            "metadata": metadata
        }
    return None

batch_size = 2**9 ###################################

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=RDBStorage(url=f"sqlite:///{DB_PATH}"),
    load_if_exists=True,
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=10),
    sampler=TPESampler(seed=seed),
    direction="minimize",
)


# Run the optimization
#study.optimize(
#    objective,
#     n_trials=30,
#    n_jobs=1,  # Use all available CPU cores for parallel trials
#)

# Print study statistics
print("\nStudy statistics: ")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

# Print details of the best trial
print("\nBest trial:")
trial = study.best_trial
print(f"Trial number: {trial.number}")
print(f"Value: {trial.value}")
print(f"Best epoch: {trial.user_attrs.get('best_epoch')}")
print("Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Evaluate the best model on the test set
evaluate_best_model(study,sav_path,batch_size)
