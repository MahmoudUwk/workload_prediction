import json
import gzip
import os
import pandas as pd
from Alibaba_helper_functions import save_object
from args import get_paths
base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()
data_file = os.path.join(base_path, 'instance_usage-000000000000.json.gz')  # Example file

all_data = []
cc = 0
with gzip.open(data_file, 'rt') as f:
    for line in f:
        cc += 1
        if cc % 100000 == 0:
            print('Line number:', cc)
        try:
            data = json.loads(line.strip())

            # Extract relevant fields, including maximum_usage.cpus
            extracted_data = {
                'start_time': data.get('start_time'),
                'sample_rate':data.get('sample_rate'),
                'end_time': data.get('end_time'),
                'machine_id': data.get('machine_id'),
                'collection_id':data.get('collection_id'),
                'average_memory_usage': data.get('average_usage', {}).get('memory'),
                'average_cpu_usage': data.get('average_usage', {}).get('cpus'),
                'maximum_cpu_usage': data.get('maximum_usage', {}).get('cpus'),
                'maximum_memory_usage': data.get('maximum_usage', {}).get('memory')# Get maximum_usage.cpus
            }
            all_data.append(extracted_data)

        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")

df = pd.DataFrame(all_data)
#%%
# Convert to numeric
numeric_cols = ['start_time', 'end_time', 'machine_id', 'collection_id','sample_rate',
                'average_cpu_usage', 'average_memory_usage', 'maximum_cpu_usage','maximum_memory_usage']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
# Calculate CPU utilization (relative to maximum_usage.cpus)
df['cpu_utilization'] = (df['average_cpu_usage'] / df['maximum_cpu_usage']) * 100
df['cpu_utilization'] = df['cpu_utilization'].fillna(0) 

df['memory_utilization'] = (df['average_memory_usage'] / df['maximum_memory_usage']) * 100
df['memory_utilization'] = df['memory_utilization'].fillna(0) 

print(df.head())
# %%
numeric_cols = ['start_time', 'end_time', 'machine_id', 'collection_id',
     'cpu_utilization','memory_utilization']


id_m = "machine_id"
df['duration'] = (df['end_time']-df['start_time'])/1e6
filtered_df = df.groupby(id_m).filter(lambda x: len(x) > 200)
filtered_df = filtered_df[filtered_df['machine_id'] != -1]
#%%
filename = os.path.join(base_path, 'google.obj')
save_object(filtered_df[numeric_cols], filename)