import pandas as pd
import os
import sys
# Assuming Alibaba_helper_functions and args_BB are defined correctly
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from Alibaba_helper_functions import loadDatasetObj, save_object
from args_BB import get_paths

# --- Configuration ---
base_path, processed_path, feat_BB_step1, feat_BB_step2, feat_BB_step3, sav_path, sav_path_plots = get_paths()
seq_length = 12
target = 'CPU usage [%]'
id_m = "machine_id"
sort_by = 'Timestamp [ms]'
k_threshold = 30
# -----------------------

# --- Load and Preprocess Data ---
df = loadDatasetObj(os.path.join(base_path, 'rnd.obj'))
df = df[['CPU usage [%]', 'machine_id', 'Timestamp [ms]']]
sav_path = feat_BB_step1  # Consider renaming this to something like 'processed_data_path'
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

# --- Filter Machines based on Average CPU Usage ---
average_values = df.groupby(id_m)[target].mean()
selected_machines = average_values[average_values > k_threshold].index.tolist()
df = df[df[id_m].isin(selected_machines)]

# --- Prepare for Grouped Operations ---
grouped = df.groupby(id_m)
#%%
# --- Process Each Machine Group ---
for M_id, M_id_val in grouped:
    sav_obj = os.path.join(sav_path, 'X_Y_M_id_' + str(M_id) + '.obj')

    if not os.path.exists(sav_obj):
        M_id_val = M_id_val.sort_values(by=[sort_by]).reset_index(drop=True)
        target_col_num = M_id_val.columns.get_loc(target)  # More efficient column index lookup

        # --- Efficient Rolling Window and Concatenation ---
        df_list = []
        y = []
        for ind in range(len(M_id_val) - seq_length):
            x = M_id_val.iloc[ind:ind + seq_length, :].copy()
            x['id'] = [(M_id,ind)]*seq_length  # More efficient way to create the ID column
            df_list.append(x)
            y.append(M_id_val.iloc[ind + seq_length, target_col_num])

        df_dataset_rolled = pd.concat(df_list, ignore_index=True)
        # -------------------------------------------------

        df_dataset_rolled = df_dataset_rolled.drop(id_m, axis=1)

        # --- Feature Extraction ---
        df_features = extract_features(df_dataset_rolled, column_id="id"
                                       , column_sort=sort_by, n_jobs=1)
        # Handle missing values
        impute(df_features)

        # --- Prepare Output ---
        df_features['M_id'] = M_id  # Efficiently add M_id column
        df_features['y'] = y
        dict_Mid = {"XY": df_features}

        # --- Save Data ---
        save_object(dict_Mid, sav_obj)

        # --- Cleanup ---
        del df_dataset_rolled
        del y
        del df_features
        del dict_Mid
        del df_list