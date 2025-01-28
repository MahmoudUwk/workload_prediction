import pandas as pd
import os
import sys
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# Assuming Alibaba_helper_functions and args_google are defined correctly

from Alibaba_helper_functions import loadDatasetObj, save_object
from args_google import get_paths
# --- Configuration ---
base_path, processed_path, feat_google_step1, feat_google_step2, feat_google_step3, sav_path, sav_path_plot = get_paths()
seq_length = 12
target = 'cpu_utilization'
id_m = "machine_id"
sort_by = 'start_time'
sav_path = feat_google_step1  # Consider renaming to something like 'processed_data_path'
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
# -----------------------
# --- Load Data ---
df = loadDatasetObj(os.path.join(base_path, 'google.obj'))[[target,id_m,sort_by]]
k_threshold = 30
# --- Filter Machines based on Average CPU Usage ---
average_values = df.groupby(id_m)[target].mean()
selected_machines = average_values[average_values > k_threshold].index.tolist()
df = df[df[id_m].isin(selected_machines)]
df = df.dropna()
filtered_df = df.groupby(id_m).filter(lambda x: len(x) > 500)
grouped = filtered_df.groupby(id_m)  # Use the filtered DataFrame
#%%
# --- Filter Data (if needed) ---
# If you need to filter for groups with more than 200 records, do it here

del df

# --- Process Each Machine Group ---
for M_id, M_id_val in grouped:
    sav_obj = os.path.join(sav_path, 'X_Y_M_id_' + str(M_id) + '.obj')

    if not os.path.exists(sav_obj):
        M_id_val = M_id_val.sort_values(by=[sort_by]).reset_index(drop=True)
        target_col_num = M_id_val.columns.get_loc(target)

        # --- Efficient Rolling Window and Concatenation ---
        df_list = []
        y = []
        for ind in range(len(M_id_val) - seq_length):
            x = M_id_val.iloc[ind:ind + seq_length, :].copy()
            x['id'] = [(M_id,ind)]*seq_length
            df_list.append(x)
            y.append(M_id_val.iloc[ind + seq_length, target_col_num])

        df_dataset_rolled = pd.concat(df_list, ignore_index=True)
        del df_list  # Free up memory
        # -------------------------------------------------

        df_dataset_rolled = df_dataset_rolled.drop(id_m, axis=1)

        # --- Feature Extraction ---
        df_features = extract_features(df_dataset_rolled, column_id="id", column_sort=sort_by, n_jobs=1)
        impute(df_features)  # Handle missing values

        # --- Prepare Output ---
        df_features['M_id'] = M_id
        df_features['y'] = y
        dict_Mid = {"XY": df_features}

        # --- Save Data ---
        save_object(dict_Mid, sav_obj)

        # --- Cleanup ---
        del df_dataset_rolled
        del y
        del df_features
        del dict_Mid