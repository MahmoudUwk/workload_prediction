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
import optuna
from optuna.storages import RDBStorage
import sqlite3
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from utils_WLP import save_object, expand_dims
from utils_WLP import get_dict_option,get_lstm_model,model_serv,expand_dims

# Function to calculate RMSE
def get_my_RMSE(test, pred):
    return np.sqrt(np.mean((np.squeeze(np.array(test)) - np.squeeze(np.array(pred))) ** 2))


flag_dataset = 2
data_set_flags = ['Alibaba','google','BB']
# Create or load the Optuna study
run_study_flag = 0
flag_load_para = 0 #1 to load the best study parameters, 0 to load the parameters manually set
N_TRIALS = 30
N_JOBS = 1
# ['InceptionTimePlus','FCNPlus', 'RNN_FCN',XceptionTimePlus,PatchTST]
model_name ='PatchTST'
from arch_config_all import get_FCNPlus,get_InceptionTimePlus,get_best_FCNPlus,get_best_InceptionTimePlus,get_RNN_FCN,get_best_RNN_FCN
from arch_config_all import get_XceptionModulePlus, get_best_XceptionModulePlus,get_ResNetPlus,get_best_ResNetPlus
from arch_config_all import get_PatchTST,get_best_PatchTST,get_para_PatchTST

select_arch = {'FCNPlus':get_FCNPlus,
 'InceptionTimePlus':get_InceptionTimePlus,
 'RNN_FCNPlus': get_RNN_FCN,
 'XceptionTimePlus':get_XceptionModulePlus,
 'ResNetPlus':get_ResNetPlus,
 'PatchTST':get_PatchTST
     }
select_arch_best = {'FCNPlus':get_best_FCNPlus,
 'InceptionTimePlus':get_best_InceptionTimePlus,
 'RNN_FCNPlus':get_best_RNN_FCN,
 'XceptionTimePlus':get_best_XceptionModulePlus,
 'ResNetPlus':get_best_ResNetPlus,
 'PatchTST':get_best_PatchTST
     }
select_arch_para = {'PatchTST':get_para_PatchTST}

arch_config_func = select_arch[model_name]
arch_config_best_func = select_arch_best[model_name]
arch_config_func_para = select_arch_para[model_name]
#%%



if data_set_flags[flag_dataset] == 'Alibaba':
    from args import get_paths
elif data_set_flags[flag_dataset]=='google':
    from args_google import get_paths
elif data_set_flags[flag_dataset]=='BB':
    from args_BB import get_paths

base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()
# Hyperparameters for the data
seq_len = 29  # Sequence length
batch_size=2**9
train_dict , val_dict, test_dict = get_dict_option(data_set_flags[flag_dataset],seq_len)
X_train = train_dict['X'][:,:,0]
y_train = train_dict['Y']
X_val = val_dict['X'][:,:,0]
y_val = val_dict['Y']
X_test = test_dict['X'][:,:,0]
y_test = test_dict['Y']
X_test_list = test_dict['X_list']
y_test_list = test_dict['Y_list']
Mids_test = test_dict['M_ids']
prediction_horizon = 1  # How many steps ahead to predict
scaler=100
X_train = np.expand_dims(np.squeeze(X_train), axis=1)
y_train = expand_dims(expand_dims(y_train))
y_val = expand_dims(expand_dims(y_val))
X_val = np.expand_dims(np.squeeze(X_val), axis=1)
X_test = np.expand_dims(np.squeeze(X_test), axis=1)

X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])

# Constants for Optuna study
STUDY_NAME = model_name+"_optimization2"
DB_PATH = os.path.join(sav_path, STUDY_NAME+".db")
MODELS_DIR = os.path.join(sav_path, STUDY_NAME+"model_FCNPlus", "models")

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





# Optuna objective function
def objective(trial):
    # Fixed parameters
    batch_size = 2**9
    n_epochs = 500  # Maximum number of epochs (early stopping will likely stop sooner)

    arch_config = arch_config_func(trial)

    # Use mixed precision and early stopping callbacks
    cbs = [EarlyStoppingCallback(monitor='valid_loss', min_delta=1e-6, patience=15)]

    # Create the TSForecaster model
    learn = TSForecaster(
        X,
        y,
        splits=splits,
        batch_size=batch_size,
        path=os.path.join(sav_path, model_name),
        arch=model_name,
        arch_config=arch_config,
        metrics=[rmse],
        cbs=cbs,
    )

    lr_to_use =   learn.lr_find().valley

    # Create a directory for the current trial to save model checkpoints
    trial_dir = os.path.join(MODELS_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    # Save trial parameters
    with open(os.path.join(trial_dir, "trial_params.json"), "w") as f:
        json.dump(trial.params, f)

    # Train the model with fit_one_cycle (or fit)
    # The EarlyStoppingCallback will stop training when validation loss stops improving
    learn.fit_one_cycle(n_epochs, lr_to_use)

    # Get the best validation loss from the recorder
    val_rmse_all = np.array(learn.recorder.values)[:,-1]
    best_epoch_index = val_rmse_all.argmin(axis=0) # Index of the best epoch
    best_val_loss = learn.recorder.values[best_epoch_index][-1]  # Best valid loss
    best_epoch = best_epoch_index + 1 # + 1 because epoch count starts from 1

    # Save the model (optional, but good practice)
    best_weights_path = os.path.join(trial_dir, "best_model.pth")
    learn.save(best_weights_path) 

    # Save trial metadata (optional)
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
def evaluate_best_model(study,arch_config_func_para=[],flag_load_para=1):
    if flag_load_para == 1: 
        best_trial = study.best_trial
        best_arch_config = arch_config_best_func(best_trial)
    else:
        best_arch_config = arch_config_func_para()
    # Create the best model
    best_model = TSForecaster(
        X,
        y,
        splits=splits,
        batch_size=batch_size,
        path=os.path.join(sav_path, "model"),
        arch=model_name,
        arch_config=best_arch_config,
        metrics=[rmse],
        cbs =  [
        EarlyStoppingCallback(monitor='valid_loss', min_delta=1e-6, patience=20),
        SaveModelCallback(monitor='valid_loss', fname='best_model', every_epoch=False)  # Save the best model
            ]
    )

    # Load the best weights
    lr_to_use =  best_model.lr_find().valley

    start_test = time.time()
    best_model.fit_one_cycle(2500, lr_to_use)##########################
    end_test = time.time()
    train_time = end_test - start_test
    # Evaluate on the test set
    y_test_preds, *_ = best_model.get_X_preds(X_test)
    y_test_preds = to_np(y_test_preds)
    test_rmse = get_my_RMSE(y_test * 100, y_test_preds * 100)
    
    y_test_pred_list = []
    rmse_list = []
    start_test = time.time()
    for c,test_sample in enumerate(X_test_list):
        
        test_sample = np.expand_dims(np.squeeze(test_sample[:,:,0]), axis=1)
        pred_i, *_ = best_model.get_X_preds(test_sample)
        pred_i = to_np(pred_i)        
    
        y_test_pred_list.append(pred_i*scaler)
        rmse_i_list = get_my_RMSE(y_test_list[c]*scaler,pred_i*scaler)
        y_test_list[c] = y_test_list[c]*scaler
        rmse_list.append(rmse_i_list)
    end_test = time.time()
    test_time = end_test - start_test
    filename = os.path.join(sav_path,model_name+'.obj')
    obj = {'y_test':y_test_list,'y_test_pred':y_test_pred_list,
           'scaler':scaler,'rmse_list':np.array(rmse_list),
           'Mids_test':Mids_test,'train_time':train_time/60,'test_time':test_time}
    save_object(obj, filename)
    
    
    
    

    return test_rmse, best_model

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

# Main execution


study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=RDBStorage(url=f"sqlite:///{DB_PATH}"),
    load_if_exists=True,
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=10),
    sampler=TPESampler(seed=seed),
    direction="minimize",
)

if run_study_flag == 1:
    # Run the optimization
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,  # Use all available CPU cores for parallel trials
    )



# Evaluate the best model on the test set
if flag_load_para == 1:
    test_rmse, best_model = evaluate_best_model(study)
else:
    test_rmse, best_model = evaluate_best_model(study,arch_config_func_para,flag_load_para)


# Save study results to CSV
df_study = study.trials_dataframe()
df_study.to_csv(os.path.join(sav_path, model_name+"_study_results.csv"), index=False)