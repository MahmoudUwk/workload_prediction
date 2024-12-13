import os
from tqdm import tqdm  # For progress bar (install with: pip install tqdm)
import pandas as pd
import gzip
base_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/google_data/'
# path_json= os.path.join(base_path,'instance_usage-000000000000.json')
# path_parquet = os.path.join(base_path,'instance_events-000000000003.parquet')
filepath_gz =   os.path.join(base_path,"instance_usage-000000000000.json.gz") # Replace with your actual file path



def process_and_store_data(filepath, chunksize=10000):
    """Processes the Google Cluster Trace data (with correct integer timestamps)."""

    all_data = []
    total_lines = 0
    file_size = os.path.getsize(filepath)

    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Processing {filepath}") as pbar:
                for chunk in pd.read_json(f, lines=True, chunksize=chunksize, convert_dates=False):
                    try:
                        processed_chunk = pd.DataFrame()

                        # Select relevant columns and create the duration column directly
                        processed_chunk = chunk.loc[:, ['start_time', 'end_time', 'machine_id', 
                        'average_usage']].copy()

                        processed_chunk['duration'] = processed_chunk['end_time'] - processed_chunk['start_time']


                        processed_chunk[['average_cpu', 'average_ram']] = processed_chunk['average_usage'].apply(pd.Series)
                        processed_chunk[['max_cpu', 'max_ram']] = processed_chunk['maximum_usage'].apply(pd.Series)


                        percentiles = pd.DataFrame(processed_chunk['cpu_usage_distribution'].tolist(), columns=[f'p{i}' for i in range(0, 101, 10)])
                        processed_chunk = pd.concat([processed_chunk, percentiles], axis=1)



                        processed_chunk = processed_chunk.drop(columns=['average_usage', 'maximum_usage', 'cpu_usage_distribution'])

                        all_data.append(processed_chunk)
                        total_lines += len(chunk)

                    except (ValueError, TypeError, KeyError) as e: # handle potential errors in chunk processing
                        print(f"Error processing chunk: {e}")

                    finally:
                        pbar.update(len(chunk.to_json(orient='records', lines=True).encode('utf-8'))) # update the progress bar


        if all_data:  # Check if any data was processed
            df = pd.concat(all_data, ignore_index=True)
            print(f"Processed {total_lines} lines.")
            return df
        else:
            print("No data appended. Check file contents or for errors in chunk processing.") # Handle cases where nothing is processed.
            return None

    except (FileNotFoundError, gzip.BadGzipFile, Exception) as e:
        print(f"Error processing file: {e}")
        return None




# Example usage (same as before)
#%%
cols = ['start_time', 'machine_id', 'average_cpu', 'average_ram']
df = process_and_store_data(filepath_gz)[cols]
# df.dropna()