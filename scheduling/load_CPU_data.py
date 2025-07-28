import os
import pickle
import pandas as pd  # Added for time series manipulation
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)  # Add logger for this module

# args.py and args_BB.py are expected to be in the Python path or same directory
from args import get_paths as get_paths_alibaba
from args_BB import get_paths as get_paths_bb

def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

def load_all_predictions_data(dataset_name):
    """
    Loads y_test, y_pred, and m_ids for all algorithms for a given dataset.

    Args:
        dataset_name (str): The name of the dataset ('Alibaba' or 'BB').

    Returns:
        dict: A dictionary where keys are algorithm display names (e.g., 'TempoSight')
              and values are dictionaries containing 'y_test', 'y_pred', 'm_ids'.
              Returns an empty dictionary if working_path is not found or other critical errors occur.
    """
    if dataset_name == 'BB':
        _, _, _, _, _, working_path, _ = get_paths_bb()
        CEDL_name = 'BB'
    elif dataset_name == 'Alibaba':
        _, _, _, _, _, working_path, _ = get_paths_alibaba()
        CEDL_name = 'Alibaba'
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: 'Alibaba', 'BB'")

    if not os.path.exists(working_path):
        print(f"Error: working_path '{working_path}' does not exist for dataset {dataset_name}.")
        return {}

    # Algorithm names as they appear in filenames or parts of filenames.
    # This list also defines the order of processing and output.
    filename_algo_parts = [
        CEDL_name + 'TST_LSTM', CEDL_name + 'EnDeAtt', CEDL_name + 'LSTM',
        'PatchTST', CEDL_name + 'CNN', 'Adaptive_predictor',
        'HistGradientBoostingRegressor', 'SVR', 'LinearRegression'
    ]

    # Mapping from filename parts to display names (used as keys in the output dict)
    # Consistent with alg_rename in plot_resultsV3.py
    algo_display_names_map = {
        CEDL_name + 'TST_LSTM': 'TempoSight',
        CEDL_name + 'EnDeAtt': 'CEDL',
        CEDL_name + 'LSTM': 'LSTM',
        'PatchTST': 'Patch\nTST', 
        CEDL_name + 'CNN': 'CNN',
        'Adaptive_predictor': 'Adaptive',
        'HistGradientBoostingRegressor': 'GBT',
        'SVR': 'SVR',
        'LinearRegression': 'LR',
    }
    
    try:
        all_files_in_working_path = os.listdir(working_path)
    except FileNotFoundError:
        print(f"Error: Could not list directory {working_path}. Path does not exist or is inaccessible.")
        return {}
        
    obj_files = [f for f in all_files_in_working_path if f.endswith(".obj")]

    loaded_data_map = {}

    for part_name in filename_algo_parts:
        target_file = None
        # Find the file that contains the algorithm part name.
        # This mimics the logic in plot_resultsV3.py for identifying algorithm-specific files.
        for fname in obj_files:
            if part_name in fname:
                target_file = fname
                break 
        
        display_name = algo_display_names_map.get(part_name)
        if not display_name:
            # This should not happen if filename_algo_parts and algo_display_names_map are consistent.
            print(f"Warning: No display name mapping for algorithm part '{part_name}'. Skipping.")
            continue

        if target_file:
            full_file_path = os.path.join(working_path, target_file)
            try:
                results_i = loadDatasetObj(full_file_path)
                loaded_data_map[display_name] = {
                    'y_test': results_i.get('y_test'),
                    'y_pred': results_i.get('y_test_pred'),
                    'm_ids': results_i.get('Mids_test')
                }
            except Exception as e:
                print(f"Error loading or processing file {full_file_path} for {display_name}: {e}")
                loaded_data_map[display_name] = {'y_test': None, 'y_pred': None, 'm_ids': None}
        else:
            print(f"Warning: Could not find a .obj file for algorithm part '{part_name}' in {working_path}")
            # Store None for missing data so the key exists, maintaining order if iterating keys later
            loaded_data_map[display_name] = {'y_test': None, 'y_pred': None, 'm_ids': None}
            
    return loaded_data_map

def inspect_specific_obj_file(obj_file_path):
    """
    Loads a specific .obj file and inspects its contents.

    Args:
        obj_file_path (str): The path to the .obj file.
    """
    print(f"\nInspecting file: {obj_file_path}")
    if not os.path.exists(obj_file_path):
        print(f"Error: File not found at {obj_file_path}")
        return

    try:
        data = loadDatasetObj(obj_file_path)
        print("Successfully loaded object.")
        if isinstance(data, dict):
            print("Keys in the object:", list(data.keys()))
            for key, value in data.items():
                print(f"\n--- Key: '{key}' ---")
                print(f"  Type: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        print(f"  First element type: {type(value[0])}")
                        if len(value) > 5:
                            print(f"  First 5 elements: {value[:5]}")
                        else:
                            print(f"  Elements: {value}")
                    else:
                        print("  Value is an empty sequence.")
                elif isinstance(value, (int, float, str, bool)):
                     print(f"  Value: {value}")
                else:
                    print("  Value: (Complex object, not displaying full content)")

        else:
            print(f"Object is not a dictionary. Type: {type(data)}")
            # Add more specific inspection if needed for non-dict types
            print(f"Data: {data}")

    except Exception as e:
        print(f"Error inspecting file {obj_file_path}: {e}")

def get_aligned_datasets(dataset_name: str, solar_obj_path: str = None, config: dict = None):
    """
    Loads CPU utilization data and solar PV data, aligns them, and returns a nested dictionary structure.
    
    Args:
        dataset_name (str): The name of the dataset ('Alibaba' or 'BB').
        solar_obj_path (str, optional): Path to solar PV data object file. If None, uses default location.
        config (dict, optional): Configuration dictionary with parameters like start_time, day_index, etc.
        
    Returns:
        dict: A nested dictionary with structure:
            {dataset_name: {algorithm_name: {machine_id: pd.DataFrame}}}
        
        Each DataFrame contains:
            - index: DatetimeIndex at 5-minute intervals
            - cpu_actual: Actual CPU utilization (%)
            - cpu_predicted: Predicted CPU utilization (%)
            - solar_pv_aligned: Solar PV generation aligned to the same timestamps, normalized to [0,1]
    """
    # Use default config if not provided
    if config is None:
        config = {
            'start_time': pd.Timestamp('2025-01-01 00:00:00'),
            'day_index': 0,  # Which day's PV data to use (0-indexed)
        }
    
    # Use default solar_obj_path if not provided
    if solar_obj_path is None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        solar_obj_path = os.path.join(current_script_dir, "data", "data_obj.obj")
    
    # Load solar PV data and upsample to 5-min intervals
    pv_series_5min = None
    try:
        solar_data_obj = loadDatasetObj(solar_obj_path)
        if 'pv_power' in solar_data_obj:
            pv_power_all_days = solar_data_obj['pv_power']
            day_idx = config.get('day_index', 0)
            if pv_power_all_days.ndim == 2 and pv_power_all_days.shape[0] > day_idx and pv_power_all_days.shape[1] == 96:
                pv_one_day = pv_power_all_days[day_idx, :]  # Use specified day's readings
                
                # Create a DatetimeIndex for the 15-min PV data
                start_time = config.get('start_time', pd.Timestamp('2025-01-01 00:00:00'))
                pv_time_index_15min = pd.date_range(start=start_time, periods=len(pv_one_day), freq='15min')
                pv_series_15min = pd.Series(pv_one_day, index=pv_time_index_15min)
                
                # Upsample to 5-minute frequency using linear interpolation
                pv_series_5min = pv_series_15min.resample('5min').interpolate(method='linear')

                if pv_series_5min is not None and not pv_series_5min.empty:
                    pv_max = pv_series_5min.max()
                    if pd.isna(pv_max) or pv_max == 0:  # Handle cases where max is NaN or zero
                        pv_series_5min = pd.Series(0.0, index=pv_series_5min.index, name=pv_series_5min.name)
                    else:
                        pv_series_5min = pv_series_5min / pv_max
                    pv_series_5min = pv_series_5min.clip(0, 1)  # Ensure it's [0,1]
                    logger.info(f"Solar PV data for {dataset_name} normalized to [0,1]. Min: {pv_series_5min.min():.2f}, Max: {pv_series_5min.max():.2f}")
                else:
                    logger.warning(f"Solar PV data for {dataset_name} is empty or None after processing. Using zeros.")
            else:
                print(f"Warning: 'pv_power' data does not have the expected shape or day_index {day_idx} is out of range.")
                print(f"Shape: {pv_power_all_days.shape}, Expected: (>={day_idx+1}, 96)")
        else:
            print(f"Warning: 'pv_power' key not found in solar data")
    except Exception as e:
        print(f"Error loading or processing solar PV data: {e}")
    
    # Load all prediction algorithm data for the specified dataset
    algo_data_map = load_all_predictions_data(dataset_name)
    if not algo_data_map:
        print(f"No data loaded for dataset {dataset_name}.")
        return {}
    
    # Create the nested result structure
    aligned_dataset = {dataset_name: {}}
    
    # Diagnostic flag to log CPU ranges only once per dataset
    cpu_range_logged_for_dataset = False

    # Process each algorithm's data
    for algo_name, data_dict in algo_data_map.items():
        # Skip if algorithm not in selected algorithms list (if provided in config)
        selected_algos = config.get('algorithms', None)
        if selected_algos and algo_name not in selected_algos:
            print(f"Skipping {algo_name} as it's not in the selected algorithms list.")
            continue
            
        # Skip if any required data is missing
        if data_dict.get('y_test') is None or data_dict.get('y_pred') is None or data_dict.get('m_ids') is None:
            print(f"Skipping {algo_name} for {dataset_name} due to missing data.")
            continue
        
        y_test = data_dict['y_test']  # Actual CPU values
        y_pred = data_dict['y_pred']  # Predicted CPU values
        m_ids = data_dict['m_ids']    # Machine IDs
        
        # Check data structure
        if not y_test or not y_pred or not m_ids:
            print(f"Empty data arrays for {algo_name}. Skipping.")
            continue
            
        # Initialize algorithm dictionary
        aligned_dataset[dataset_name][algo_name] = {}
        
        # Process each machine's data
        for idx, m_id in enumerate(m_ids):
            # Get this machine's CPU utilization data
            if isinstance(y_test, np.ndarray) and y_test.ndim > 1:
                # Handle multidimensional arrays - each row is a machine's data
                actual_cpu = y_test[idx]
                pred_cpu = y_pred[idx]
            elif isinstance(y_test[0], (list, np.ndarray)):
                # If y_test is a list of lists (one sublist per machine)
                actual_cpu = y_test[idx]
                pred_cpu = y_pred[idx]
            else:
                # If there's only one machine's data
                actual_cpu = y_test
                pred_cpu = y_pred
            
            # Create DatetimeIndex for CPU data (5-min interval)
            start_time = config.get('start_time', pd.Timestamp('2025-01-01 00:00:00'))
            cpu_time_index = pd.date_range(start=start_time, periods=len(actual_cpu), freq='5min')
            
            # Handle arrays that might be multi-dimensional
            if isinstance(actual_cpu, np.ndarray) and actual_cpu.ndim > 1:
                actual_cpu = actual_cpu.flatten()
            if isinstance(pred_cpu, np.ndarray) and pred_cpu.ndim > 1:
                pred_cpu = pred_cpu.flatten()
            
            # Create DataFrame with CPU data
            machine_df = pd.DataFrame({
                'cpu_actual': actual_cpu,
                'cpu_predicted': pred_cpu
            }, index=cpu_time_index)
            
            # Ensure CPU values are in percentage (0-100)
            if machine_df['cpu_actual'].max() <= 1.0:
                machine_df['cpu_actual'] *= 100.0
            if machine_df['cpu_predicted'].max() <= 1.0:
                machine_df['cpu_predicted'] *= 100.0
                
            # Align solar PV data with this machine's time index
            if pv_series_5min is not None:
                # Reindex and interpolate to match machine's time index
                pv_aligned = pv_series_5min.reindex(machine_df.index).interpolate(method='linear')
                machine_df['solar_pv_aligned'] = pv_aligned
            else:
                # If no solar data, add placeholder column of zeros
                machine_df['solar_pv_aligned'] = 0.0
                logger.warning(f"Using all zeros for solar_pv_aligned for {dataset_name}, {algo_name}, {m_id} as PV data was not available.")
            
            # --- CPU Data Range Logging ---
            if not cpu_range_logged_for_dataset:
                try:
                    actual_min = np.nanmin(machine_df['cpu_actual'])
                    actual_max = np.nanmax(machine_df['cpu_actual'])
                    pred_min = np.nanmin(machine_df['cpu_predicted'])
                    pred_max = np.nanmax(machine_df['cpu_predicted'])
                    logger.info(f"CPU Data Range Check for {dataset_name} (first machine {m_id}, algo {algo_name}):")
                    logger.info(f"  cpu_actual: Min={actual_min:.2f}, Max={actual_max:.2f}")
                    logger.info(f"  cpu_predicted: Min={pred_min:.2f}, Max={pred_max:.2f}")
                    cpu_range_logged_for_dataset = True  # Log only once
                except KeyError as e:
                    logger.warning(f"Could not log CPU ranges for {m_id}, missing column: {e}")
                except Exception as e:
                    logger.warning(f"Error during CPU range logging for {m_id}: {e}")
            # --- End CPU Data Range Logging ---
            
            # Store DataFrame in the nested dictionary
            aligned_dataset[dataset_name][algo_name][m_id] = machine_df
    
    return aligned_dataset

if __name__ == '__main__':
    # Example usage:
    # This block will run if the script is executed directly.
    # It assumes that args.py, args_BB.py, and utils_WLP.py are accessible.

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load solar PV data from data_obj.obj
    solar_obj_file_path = os.path.join(current_script_dir, "data", "data_obj.obj")
    print(f"\nAttempting to load solar PV data from: {solar_obj_file_path}")
    pv_series_5min_aligned = None
    try:
        solar_data_obj = loadDatasetObj(solar_obj_file_path)
        if 'pv_power' in solar_data_obj:
            pv_power_all_days = solar_data_obj['pv_power']
            # Assuming pv_power_all_days is (num_days, 96 readings per day)
            # Using the first day's PV data
            if pv_power_all_days.ndim == 2 and pv_power_all_days.shape[0] > 0 and pv_power_all_days.shape[1] == 96:
                pv_one_day = pv_power_all_days[0, :] # First day, 96 readings
                print(f"Successfully loaded 'pv_power' data. Using first day's {len(pv_one_day)} readings (15-min interval).")

                # Create a DatetimeIndex for the 15-min PV data
                # Using an arbitrary start date for demonstration
                start_time = pd.Timestamp('2025-01-01 00:00:00')
                pv_time_index_15min = pd.date_range(start=start_time, periods=len(pv_one_day), freq='15min')
                pv_series_15min = pd.Series(pv_one_day, index=pv_time_index_15min)

                # Upsample to 5-minute frequency using linear interpolation
                pv_series_5min = pv_series_15min.resample('5min').interpolate(method='linear')
                print(f"PV data upsampled to 5-min interval. Length: {len(pv_series_5min)} points.")
                # This series can now be aligned with CPU data of varying lengths

            else:
                print(f"Warning: 'pv_power' data in {solar_obj_file_path} does not have the expected shape (days, 96). Found shape: {pv_power_all_days.shape}")
        else:
            print(f"Warning: 'pv_power' key not found in {solar_obj_file_path}")
    except Exception as e:
        print(f"Error loading or processing solar PV data from {solar_obj_file_path}: {e}")

    print("\n" + "="*50 + "\n")

    print("Attempting to load data for Alibaba dataset...")
    alibaba_data_results = load_all_predictions_data('Alibaba')
    if alibaba_data_results:
        # Demonstrate alignment with the first algorithm's y_test data
        if pv_series_5min is not None: # Check if PV data was loaded and upsampled successfully
            first_alg_name = list(alibaba_data_results.keys())[0]
            data_dict = alibaba_data_results[first_alg_name]
            
            print(f"\nSynchronizing PV data with CPU data for Algorithm: {first_alg_name}")

            if data_dict.get('y_test') is not None:
                cpu_y_test_data = data_dict['y_test']
                
                # Assuming y_test is a flat list of CPU utilization values based on prior script output
                # If y_test is a list of lists (per machine), you'd select one: cpu_y_test_data = data_dict['y_test'][0]
                if isinstance(cpu_y_test_data, list) and len(cpu_y_test_data) > 0:
                    # Create DatetimeIndex for CPU data (5-min interval)
                    # Ensure it uses the same start_time for direct comparison if desired, or adjust as needed
                    cpu_time_index_5min = pd.date_range(start=start_time, periods=len(cpu_y_test_data), freq='5min')
                    cpu_series_5min = pd.Series(cpu_y_test_data, index=cpu_time_index_5min)
                    
                    # Align (reindex and interpolate) pv_series_5min to match cpu_series_5min's index
                    # This handles cases where CPU data might be shorter or longer than one day's upsampled PV data
                    pv_series_5min_aligned = pv_series_5min.reindex(cpu_series_5min.index).interpolate(method='linear')
                    # Fill any remaining NaNs at the beginning (if cpu_series starts before pv_series) or end
                    #pv_series_5min_aligned = pv_series_5min_aligned.fillna(method='bfill').fillna(method='ffill')


                    print(f"  CPU data ('y_test' for {first_alg_name}):")
                    print(f"    Length: {len(cpu_series_5min)} points (5-min interval)")
                    print(f"    First 5 values: {cpu_series_5min.head().tolist()}")
                    
                    print(f"  Synchronized PV data (aligned to CPU data):")
                    print(f"    Length: {len(pv_series_5min_aligned)} points (5-min interval)")
                    print(f"    First 5 values: {pv_series_5min_aligned.head().tolist()}")
                    if pv_series_5min_aligned.isnull().any():
                        print(f"    Warning: Synchronized PV data contains NaNs: {pv_series_5min_aligned.isnull().sum()} NaN(s)")

                else:
                    print(f"  'y_test' for {first_alg_name} is empty or not a list.")
            else:
                print(f"  'y_test' not found for {first_alg_name}.")
        else:
            print("\nPV data not available for synchronization.")

        for alg_name, data_dict_loop in alibaba_data_results.items():
            print(f"\nAlgorithm: {alg_name}")
            if data_dict_loop.get('y_test') is not None and data_dict_loop.get('y_pred') is not None:
                y_test_len = len(data_dict_loop['y_test']) if data_dict_loop['y_test'] is not None else 'N/A'
                y_pred_len = len(data_dict_loop['y_pred']) if data_dict_loop['y_pred'] is not None else 'N/A'
                m_ids_len = len(data_dict_loop['m_ids']) if data_dict_loop['m_ids'] is not None else 'N/A'
                print(f"  y_test items: {y_test_len}")
                print(f"  y_pred items: {y_pred_len}")
                print(f"  m_ids items: {m_ids_len}")
            else:
                print("  Data not fully loaded (y_test, y_pred, or m_ids might be None or missing).")
    else:
        print("No data loaded for Alibaba dataset. Check paths, file availability, and previous warnings.")

    print("\n" + "="*50 + "\n")

    print("Attempting to load data for BB (Bitbrains) dataset...")
    bb_data_results = load_all_predictions_data('BB')
    if bb_data_results:
        for alg_name, data_dict in bb_data_results.items():
            print(f"\nAlgorithm: {alg_name}")
            if data_dict.get('y_test') is not None and data_dict.get('y_pred') is not None:
                y_test_len = len(data_dict['y_test']) if data_dict['y_test'] is not None else 'N/A'
                y_pred_len = len(data_dict['y_pred']) if data_dict['y_pred'] is not None else 'N/A'
                m_ids_len = len(data_dict['m_ids']) if data_dict['m_ids'] is not None else 'N/A'
                print(f"  y_test items: {y_test_len}")
                print(f"  y_pred items: {y_pred_len}")
                print(f"  m_ids items: {m_ids_len}")
            else:
                print("  Data not fully loaded (y_test, y_pred, or m_ids might be None or missing).")
    else:
        print("No data loaded for BB dataset. Check paths, file availability, and previous warnings.")
