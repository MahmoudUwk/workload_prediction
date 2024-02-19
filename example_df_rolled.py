# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:09:16 2024

@author: mahmo
"""

import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import roll_time_series

df = pd.DataFrame({
    "id": [1, 1, 1, 1, 2, 2],
    "time": [1, 2, 3, 4, 8, 9],
    "x": [20,21,22,23,24,25]})#,
#     "y": [5, 6, 7, 8, 12, 13],
# })
# df = df.set_index('id')
seq = 3
# df_rolled = roll_time_series(df, column_id="id", column_sort="time",n_jobs = 1,max_timeshift=seq,min_timeshift=seq)

# df_features = extract_features(df_rolled, column_id="id", column_sort="time",n_jobs = 1)

df2,y = make_forecasting_frame(df,kind="x",min_timeshift = seq,max_timeshift=seq,rolling_direction=1)

#(x, kind,min_timeshift, max_timeshift, rolling_direction)

# from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
#     load_robot_execution_failures
# download_robot_execution_failures()
# timeseries, y = load_robot_execution_failures()

# print(timeseries.head())

# from tsfresh import extract_features
# extracted_features = extract_features(timeseries, column_id="id", column_sort="time",n_jobs = 1)












