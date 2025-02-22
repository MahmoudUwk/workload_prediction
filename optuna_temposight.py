import optuna
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils_WLP import TempoSight
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from utils_WLP import log_and_save, expand_dims,get_dict_option,RMSE,MAPE,MAE
import time
import os

scaler = 100
num_epoc = 700
seq = 32
#%%
dataset_flag = 1

batch_size = 2**9

from args import get_paths
dataset =['Alibaba','BB'][dataset_flag]
version = 'TempoSight_V1'+dataset
base_path,_,_,_,_,sav_path,_ = get_paths()
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
        
train_dict , val_dict, test_dict = get_dict_option(dataset,seq)


X_train = train_dict['X']
y_train = train_dict['Y']

X_val = val_dict['X']
y_val = val_dict['Y']

X_test_list = test_dict['X_list']
y_test_list = test_dict['Y_list']
X_test = test_dict['X']
y_test = test_dict['Y']
Mids_test = test_dict['M_ids']

output_dim = 1
input_dim=(X_train.shape[1],X_train.shape[2])
y_train = expand_dims(expand_dims(y_train))
y_val = expand_dims(expand_dims(y_val))


#%%

# === Objective Function for Optuna ===
def objective(trial):
    data_set = 'Alibaba'
    model_name = 'Exp_patch_TST_LSTM_learnable'+version
    # Hyperparameter suggestions
    bottle_neck =  trial.suggest_categorical("bottle_neck", [8,32,64])
    fusion_dim  = trial.suggest_categorical("fusion_dim", [64,128,256])
    patch_length = 16#trial.suggest_categorical("patch_length", [8,16])
    lstm_units = trial.suggest_categorical("lstm_units", [128,256])
    lstm_layers = trial.suggest_int("lstm_layers", 2, 4)
    num_heads = trial.suggest_categorical("num_heads", [4,8,16])
    transformer_ff_dim =  trial.suggest_categorical("transformer_ff_dim", [128,256,512])
    num_transformer_layers = trial.suggest_int("num_transformer_layers", 3, 5)
    dropout_rate = trial.suggest_categorical("dropout_rate", [0,0.2,0.3])

    # Build the model with the current hyperparameters
    model = TempoSight(
        input_shape=input_dim,
        pred_len=output_dim,
        # patch_length=patch_length,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        num_heads=num_heads,
        transformer_ff_dim=transformer_ff_dim,
        num_transformer_layers=num_transformer_layers,
        bottle_neck = bottle_neck,
        fusion_dim=fusion_dim,
        dropout_rate=dropout_rate,
    )
    
    # Define callbacks for early stopping (to prevent overfitting and speed up optimization)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8
                                  , patience=5, min_lr=3e-4)
    callbacks = [EarlyStopping(monitor='val_loss', 
                        patience=20, restore_best_weights=True),reduce_lr]
    start_train = time.time()
    # Train the model. Adjust epochs and batch_size as needed.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=600,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    end_train = time.time()
    train_time = (end_train - start_train)/60
    
    y_test_pred = (model.predict(X_test))*scaler
    row_alibaba = [RMSE(y_test*scaler,y_test_pred),MAE(y_test*scaler,y_test_pred)
                   ,MAPE(y_test*scaler,y_test_pred)]
    # Use the best validation loss as the objective to minimize
    best_val_loss = min(history.history["val_loss"])
    row_log = row_alibaba+[patch_length,lstm_units,lstm_layers,num_heads,transformer_ff_dim,
     num_transformer_layers,dropout_rate,fusion_dim,bottle_neck]
    cols =  ['RMSE','MAE','MAPE','patch_length','lstm_units','lstm_layers','num_heads',
             'transformer_ff_dim','num_transformer_layers','dropout_rate',
             'fusion_dim','bottle_neck']
    log_and_save(row_log,cols,data_set,model_name,sav_path,
                 X_test_list,y_test_list,model,batch_size,scaler,train_time,Mids_test)
    print(row_log)
    return best_val_loss

# === Run Optuna Study ===
if __name__ == "__main__":
    storage_name = "sqlite:///optuna_study_learable"+version+".db"
    study = optuna.create_study(
        direction="minimize",
        storage=storage_name,
        study_name="parallel_lstm_patchtst_optimized_learable"+version,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=600, n_jobs=1)


    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
