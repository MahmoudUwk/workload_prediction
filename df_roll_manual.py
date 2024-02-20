
import pandas as pd
import numpy as np
import pickle
import os
from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import roll_time_series
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    
base_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/"
# base_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/"
script = "server_usage.csv"
target = " used percent of cpus(%)"
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None

info_path = base_path+"schema.csv"

df_info =  pd.read_csv(info_path)
df_info = df_info[df_info["file name"] == script]['content']
#%%

full_path = base_path+script
nrows = None
df =  pd.read_csv(full_path,nrows=nrows,header=None,names=list(df_info))

df = df[[" machine id", " timestamp"," used percent of cpus(%)"]]
# df = df[df.notna()]
df = df.dropna()
seq_length = 12


sav_path = base_path+"feature_obj"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
#%%
grouped = df.groupby([" machine id"])
dataset_widnows = []
M_ids = []
label_pred = []
# ids_rolled = []


for M_id, M_id_val in grouped:
    df_dataset_rolled = pd.DataFrame(columns=list(df.columns)+['id'])
    y = []
    M_id_val = M_id_val.sort_values(by=[' timestamp']).reset_index(drop=True)
    print(M_id)
    for ind in range(len(M_id_val)-seq_length):
        x = M_id_val.loc[ind:ind+seq_length-1].copy()
        x['id'] = [(M_id[0],ind)]*seq_length
        if ind == 0:
            df_dataset_rolled = x.copy()     
        else:
            df_dataset_rolled = pd.concat([df_dataset_rolled, x], ignore_index=True)
        
        
        y.append(M_id_val[target][ind+seq_length-1:ind+seq_length].iloc[0])
        
        
    
    df_dataset_rolled = df_dataset_rolled.drop([' machine id'], axis=1)
    df_features = extract_features(df_dataset_rolled, column_id="id", column_sort=" timestamp",n_jobs = 1).dropna(axis=1, how='all')
    df_features['M_id'] = [M_id[0]]*len(df_features)
    # df_features = df_features.set_index('M_id')
    dict_Mid = {"X":df_features,"y":y}
    assert(len(y)==len(df_features))
    save_object(dict_Mid, os.path.join(sav_path,'X_Y_M_id_'+str(M_id[0])+'.obj'))










