from google.cloud import storage
from args import get_paths
import os
from google.auth import default

# Explicitly get credentials
credentials, project_id = default()

# Create the storage client with the credentials
storage_client = storage.Client(credentials=credentials, project=project_id)
base_path,processed_path,_,_,feat_stats_step3 = get_paths()
# Initialize a Cloud Storage client
client = storage.Client()

# Specify the bucket and file pattern
bucket_name = "clusterdata_2019_a"
file_prefix = "instance_usage-"
local_directory = os.path.join(base_path,'google_data') # Make sure the directory exists
if not os.path.exists(local_directory):
    os.makedirs(local_directory)

# Get the bucket
bucket = client.bucket(bucket_name)

# List and download blobs matching the prefix
blobs = bucket.list_blobs(prefix=file_prefix)
for blob in blobs:
    local_path = f"{local_directory}/{blob.name}"
    print(f"Downloading {blob.name} to {local_path}")
    blob.download_to_filename(local_path)

    # Decompress using gzip (or other tools if needed)
    import gzip
    with gzip.open(local_path, 'rb') as f_in:
        with open(local_path[:-3], 'wb') as f_out:  # Remove .gz extension
            f_out.write(f_in.read())
    print(f"Decompressed to {local_path[:-3]}")
    # Optionally delete the .gz file
    os.remove(local_path) 