import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from Alibaba_helper_functions import loadDatasetObj, save_object, MAPE, MAE, RMSE, get_lstm_model
from Alibaba_helper_functions import expand_dims, list_to_array, get_EN_DE_LSTM_model, get_en_de_lstm_model_attention
from Alibaba_fet_features_LSTM_no_cluster import get_dataset_alibaba_lstm_no_cluster
import warnings
from keras.callbacks import ReduceLROnPlateau
import optuna
import time
import tensorflow as tf

# warnings.filterwarnings('ignore')
# np.random.seed(7)

from Alibaba_helper_functions import get_google_data, get_BB_data

# X_BB.Y_BB = get_BB_data(seq,scaler)
# X_google.Y_google = get_google_data(seq,scaler)
def log_results_LSTM(row, save_path, save_name):
    # save_name = 'results_LSTM_en_de_HP_seach.csv'
    cols = ["RMSE", "MAE", "MAPE(%)", "seq", "num_layers", "units"]
    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path, save_name)):
        df3.to_csv(os.path.join(save_path, save_name), index=False)

    df = pd.read_csv(os.path.join(save_path, save_name))
    df.loc[len(df)] = row
    flag = 0
    if len(df) != 0:
        if row[0] == df.min()['RMSE']:
            flag = 1
    else:
        flag = 1
    df.to_csv(os.path.join(save_path, save_name), mode='w', index=False, header=True)
    return flag

# %%
def get_data(num_feat, params):
    X_train, y_train, X_val, y_val, X_test_list, y_test_list, scaler, Mids_test = get_dataset_alibaba_lstm_no_cluster(
        params['seq'], num_feat)
    X_test = list_to_array(X_test_list, params['seq'], num_feat)
    y_test = list_to_array(y_test_list, 0, num_feat)
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, X_test_list, y_test_list, Mids_test

def objective(trial, num_feat, num_epoc, batch_size, save_path):
    # Suggest hyperparameters
    units = trial.suggest_categorical('units', [64, 128, 256,512,1024])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    seq = trial.suggest_int('seq', 1, 32)
    dense_units = trial.suggest_categorical('dense_units', [128, 256, 512,1024])
    # lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    params = {
        'units': units,
        'num_layers': num_layers,
        'seq': seq,
        # 'lr': lr,
        'dense_units': dense_units
    }

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, X_test_list, y_test_list, Mids_test = get_data(num_feat,
                                                                                                          params)
    X_BB, Y_BB = get_BB_data(params['seq'], scaler)
    X_google, Y_google = get_google_data(params['seq'], scaler)

    output_dim = 1
    input_dim = (X_train.shape[1], X_train.shape[2])
    y_train = expand_dims(expand_dims(y_train))
    y_val = expand_dims(expand_dims(y_val))
    y_test = expand_dims(expand_dims(y_test))

    # model = get_EN_DE_LSTM_model(input_dim,output_dim,**params)
    model = get_en_de_lstm_model_attention(input_dim, output_dim, **params)
    # model = get_lstm_model(input_dim,output_dim,**params)
    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)

    callbacks_list = [EarlyStopping(monitor='val_loss',
                                    patience=15, restore_best_weights=True)]  # ,lr_scheduler]
    
    model.fit(X_train, y_train, epochs=num_epoc,
              batch_size=batch_size, verbose=0, shuffle=True,
              validation_data=(X_val, y_val), callbacks=callbacks_list)
    
    y_test_pred = (model.predict(X_test)) * scaler
    row_alibaba = [RMSE(y_test * scaler, y_test_pred), MAE(y_test * scaler, y_test_pred)
                       , MAPE(y_test * scaler, y_test_pred)] + [params['seq'], params['num_layers'], params['units']]
    flag_ali = log_results_LSTM(row_alibaba, save_path, 'CEDL_Alibaba.csv')
    if flag_ali == 1:
        model.export(os.path.join(save_path, 'Alibaba'))

    y_pred_BB = (model.predict(X_BB, verbose=0)) * scaler
    row_BB = [RMSE(Y_BB, y_pred_BB), MAE(Y_BB, y_pred_BB),
              MAPE(Y_BB, y_pred_BB)] + [params['seq'], params['num_layers'], params['units']]
    flag_BB = log_results_LSTM(row_BB, save_path, 'CEDL_BB.csv')
    if flag_BB == 1:
        model.export(os.path.join(save_path, 'BB'))

    y_pred_google = (model.predict(X_google, verbose=0)) * scaler
    row_google = [RMSE(Y_google, y_pred_google), MAE(Y_google, y_pred_google)
                      , MAPE(Y_google, y_pred_google)] + [params['seq'], params['num_layers'], params['units']]
    flag_google = log_results_LSTM(row_google, save_path, 'CEDL_google.csv')
    if flag_google == 1:
        model.export(os.path.join(save_path, 'Google'))
    print("RMSE:", row_alibaba[0])
    return row_alibaba[0]

# %%
from args import get_paths
study_name = 'LSTM_HP_Optimization_val'
_, _, _, _, _, sav_path, _ = get_paths()
sav_path = os.path.join(sav_path,study_name)
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
num_feat = 2
run_search = 1
num_epoc = 2500
batch_size = 2 ** 9
n_trials = 80

# %%
if run_search:
    # Create a study object and optimize the objective function.
    study = optuna.create_study(study_name=study_name, direction='minimize')
    study.optimize(lambda trial: objective(trial, num_feat, num_epoc, batch_size, sav_path), n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Save study results to CSV
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(sav_path, study_name + '.csv'))