
import pandas as pd
import numpy as np
import pickle
import os
from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import roll_time_series
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
script = "server_usage.csv"
target = " used percent of cpus(%)"
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None

info_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/schema.csv"

df_info =  pd.read_csv(info_path)
df_info = df_info[df_info["file name"] == script]['content']
#%%
base_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/"
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

y = []
for M_id, M_id_val in grouped:
    df_dataset_rolled = pd.DataFrame(columns=list(df.columns)+['id'])
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
    save_object(df_features, os.path.join(sav_path,'df_features_M_id_'+str(M_id)+'.obj'))



#     label_pred.append(y)
#     dataset_widnows.append(x)
#     M_ids.append([M_id[0]]*len(y))

# M_ids = [ x for xs in M_ids for x in xs]
# dataset_widnows = list_to_array(dataset_widnows,seq)

# label_pred = list_to_array(label_pred,0)









