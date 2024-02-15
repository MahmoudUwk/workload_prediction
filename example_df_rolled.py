# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:09:16 2024

@author: mahmo
"""

import pandas as pd
df = pd.DataFrame({
    "id": [1, 1, 1, 1, 2, 2],
    "time": [1, 2, 3, 4, 8, 9],
    "x": [1, 2, 3, 4, 10, 11],
    "y": [5, 6, 7, 8, 12, 13],
})

from tsfresh.utilities.dataframe_functions import roll_time_series
df_rolled = roll_time_series(df, column_id="id", column_sort="time",n_jobs = 1)
from tsfresh import extract_features
df_features = extract_features(df_rolled, column_id="id", column_sort="time",n_jobs = 1)
# from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
#     load_robot_execution_failures
# download_robot_execution_failures()
# timeseries, y = load_robot_execution_failures()
# #%%
# from tsfresh import extract_features
# extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
#%%
# from tsfresh.examples.robot_execution_failures import \
#     download_robot_execution_failures, \
#     load_robot_execution_failures
# from tsfresh.feature_extraction import extract_features
# from tsfresh.utilities.distribution import MultiprocessingDistributor

# # download and load some time series data
# download_robot_execution_failures()
# df, y = load_robot_execution_failures()

# # We construct a Distributor that will spawn the calculations
# # over four threads on the local machine
# Distributor = MultiprocessingDistributor(n_workers=1,
#                                           disable_progressbar=False,
#                                           progressbar_title="Feature Extraction")

# # just to pass the Distributor object to
# # the feature extraction, along with the other parameters
# X = extract_features(timeseries_container=df,column_id='id',column_sort='time',n_jobs = 1)

















