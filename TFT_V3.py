import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss,RMSE
import pandas as pd
import numpy as np
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle
from lightning.pytorch.loggers import TensorBoardLogger
import torch

torch.set_float32_matmul_precision('medium')

# Assuming these are your helper functions (you might need to adjust them)
from Alibaba_helper_functions import get_train_test_Mids, save_object

def get_time_data(df,max_encoder_length,max_prediction_length):
    X_TD = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="cpu_util",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],  # Add static categoricals if you have any
        static_reals=[],  # Add static reals if you have any
        time_varying_known_reals=["time_idx"],  # Add time-varying known reals
        time_varying_unknown_reals=[
            "cpu_util",
            "mem_util",
        ],  # Add time-varying unknown reals
        target_normalizer=None,
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
    )
    return X_TD

# 1. Data Loading and Preprocessing Using get_df_tft
def prepare_data_for_tft(df, sequence_length, prediction_horizon):
    df.dropna(inplace=True)
    df['timestamp'] = df[' timestamp'].astype(str).str.strip()
    # Convert to datetime, handling any invalid parsing gracefully
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values([' machine id', 'timestamp'])
    df['time_idx'] = df.groupby(' machine id').cumcount()

    df.rename(
        columns={
            ' used percent of cpus(%)': "cpu_util",
            ' used percent of memory(%)': "mem_util",
            ' machine id': "group_id",
        },
        inplace=True,
    )
    # You might need to add or modify features here

    cols = ['time_idx', 'group_id', 'cpu_util', 'mem_util']
    return df.loc[:, cols]

def get_df_tft(seq_length):
    from args import get_paths

    base_path, _, _, _, _, _, _ = get_paths()

    script = "server_usage.csv"
    info_path = base_path + "schema.csv"
    normalize_cols = [' used percent of cpus(%)', ' used percent of memory(%)']
    cols = [
        ' timestamp',
        ' used percent of cpus(%)',
        ' used percent of memory(%)',
        ' machine id',
    ]
    df_info = pd.read_csv(info_path)
    df_info = df_info[df_info["file name"] == script]['content']

    full_path = base_path + script
    nrows = None
    df = pd.read_csv(
        full_path, nrows=nrows, header=None, names=list(df_info)
    )[cols]
    scaler = 100
    df.loc[:, normalize_cols] = df.loc[:, normalize_cols] / scaler
    df = df.dropna()

    train_val_per = 0.8
    val_per = 0.16

    M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(
        base_path, train_val_per, val_per
    )

    df_train = df_from_M_id(df, M_ids_train)
    df_val = df_from_M_id(df, M_ids_val)
    df_test = df_from_M_id(df, M_ids_test)

    return df_train, df_val, df_test

def df_from_M_id(df, M_id_list):
    df = df[df[' machine id'].isin(M_id_list)]
    return df

# Filter out short groups
def filter_short_groups(df, min_length, group_col="group_id"):
    group_lengths = df.groupby(group_col).size()
    long_groups = group_lengths[group_lengths >= min_length].index
    return df[df[group_col].isin(long_groups)]

# Example Usage (replace with your actual data loading logic)
sequence_length = 32  # Adjust as needed
prediction_horizon = 1  # Adjust as needed

scaler = 100
df_train, df_val, df_test = get_df_tft(sequence_length)

# Prepare data for TFT
train_df = prepare_data_for_tft(
    df_train, sequence_length, prediction_horizon
)
train_df.reset_index(drop=True, inplace=True)
val_df = prepare_data_for_tft(
    df_val, sequence_length, prediction_horizon
)
val_df.reset_index(drop=True, inplace=True)
test_df = prepare_data_for_tft(
    df_test, sequence_length, prediction_horizon
)
test_df.reset_index(drop=True, inplace=True)

# Filter out short groups
test_df = filter_short_groups(
    test_df, sequence_length + prediction_horizon
)

del df_train, df_val, df_test

# 2. Create TimeSeriesDataSet
max_prediction_length = prediction_horizon
max_encoder_length = sequence_length

training = get_time_data(train_df,max_encoder_length,max_prediction_length)
validation = get_time_data(val_df,max_encoder_length,max_prediction_length)
testt = get_time_data(test_df,max_encoder_length,max_prediction_length)

batch_size = 128  # Adjust based on your GPU memory and experimentation
test_dataloader = testt.to_dataloader(
    train=False, batch_size=batch_size, num_workers=10
)
# 3. Create DataLoaders
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=8
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=8
)


#
logger = TensorBoardLogger("tb_logs3", name="my_model3")
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-5,
    patience=20,
    verbose=True,
    mode="min",
)
lr_logger = pl.callbacks.LearningRateMonitor()

# Create a new trainer for the final training
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.5,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.002,
    lstm_layers =1,
    hidden_size=64,
    attention_head_size=4,
    dropout=0,
    hidden_continuous_size=32,
    loss=RMSE(),
    log_interval=1,
    log_val_interval=1,
    #reduce_on_plateau_patience=3,
)



# # make a prediction on entire validation set
# preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)


# Train the final model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# 6. Evaluate the Model on the Test Set
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Best model path: {best_model_path}")

predictions = tft.predict(
    test_dataloader, batch_size=batch_size
)
y_pred = predictions.to('cpu').numpy()
y_test = np.concatenate([y[0] for x, y in iter(test_dataloader)])

# Calculate metrics
rmse = np.sqrt(np.mean((y_pred * 100 - y_test * 100) ** 2))
mae = np.mean(np.abs(y_pred * 100 - y_test * 100))
mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")