"""
Metrics calculator for evaluating the performance of different scheduling policies.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import math
from src.simulation_engine import SimulationResult

def calculate_simulation_metrics(simulation_result: SimulationResult, 
                                machine_df: pd.DataFrame, 
                                workload_cfg: Dict[str, Any], 
                                power_model: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate various metrics for a simulation run.
    
    Args:
        simulation_result: The result of a scheduling simulation
        machine_df: DataFrame with cpu_actual, cpu_predicted, solar_pv_aligned data
        workload_cfg: Workload configuration
        power_model: Power consumption model
        
    Returns:
        Dictionary of calculated metrics
    """
    # Check if the simulation was successful
    if not simulation_result.scheduling_success:
        return {
            "energy_consumed_wh": None,
            "green_energy_wh": None,
            "grid_energy_wh": None,
            "geuf_percentage": None,
            "scheduling_success": False,
            "failure_reason": simulation_result.reason_if_failed
        }
    
    # Get the window data from the simulation result
    start_time = simulation_result.start_time
    end_time = simulation_result.end_time
    window_data = machine_df.loc[start_time:end_time].copy()
    
    # Duration of each time step in hours
    time_step = (window_data.index[1] - window_data.index[0]).total_seconds() / 3600  # Hours
    
    # Calculate estimated power consumption at each time step based on CPU utilization
    normalized_cpu = window_data['cpu_actual'] / 100.0
    cores_used = workload_cfg['cores_required']
    
    # Calculate power consumption based on the specified curve
    if power_model.get('utilization_to_power_curve') == 'quadratic':
        # Quadratic relationship between CPU and power
        window_data['power_watts'] = (
            power_model['idle_power_watts'] + 
            (power_model['power_per_core_watts'] * cores_used * normalized_cpu**2)
        )
    else:
        # Linear relationship between CPU and power
        window_data['power_watts'] = (
            power_model['idle_power_watts'] + 
            (power_model['power_per_core_watts'] * cores_used * normalized_cpu)
        )
    
    # Calculate energy consumed at each time step (watt-hours)
    window_data['energy_wh'] = window_data['power_watts'] * time_step
    
    # Calculate total energy consumption
    total_energy_consumed = window_data['energy_wh'].sum()
    
    # Normalize solar PV data to match the energy scale (depends on your solar data format)
    # Assuming the max solar generation can meet the max power demand
    # This is a simplified model - adjust based on your solar data calibration
    max_power = power_model['idle_power_watts'] + power_model['power_per_core_watts'] * cores_used
    max_solar = window_data['solar_pv_aligned'].max()
    if max_solar > 0:
        solar_scale_factor = max_power / max_solar
    else:
        solar_scale_factor = 1.0
    
    # Calculate available green energy at each time step (watt-hours)
    window_data['green_energy_wh'] = window_data['solar_pv_aligned'] * solar_scale_factor * time_step
    
    # Cap green energy used to actual energy consumed at each time step
    window_data['green_energy_used_wh'] = window_data[['energy_wh', 'green_energy_wh']].min(axis=1)
    
    # Calculate grid energy used (total - green)
    window_data['grid_energy_wh'] = window_data['energy_wh'] - window_data['green_energy_used_wh']
    
    # Ensure grid energy is not negative
    window_data['grid_energy_wh'] = window_data['grid_energy_wh'].clip(lower=0)
    
    # Calculate total green and grid energy
    total_green_energy = window_data['green_energy_used_wh'].sum()
    total_grid_energy = window_data['grid_energy_wh'].sum()
    
    # Calculate GEUF (Green Energy Utilization Factor)
    if total_energy_consumed > 0:
        geuf_percentage = (total_green_energy / total_energy_consumed) * 100
    else:
        geuf_percentage = 0.0
    
    # Return all calculated metrics
    return {
        "energy_consumed_wh": total_energy_consumed,
        "green_energy_wh": total_green_energy,
        "grid_energy_wh": total_grid_energy,
        "geuf_percentage": geuf_percentage,
        "scheduling_success": True,
        "failure_reason": None
    }

def calculate_prediction_accuracy(actual_series: pd.Series, predicted_series: pd.Series) -> Dict[str, float]:
    """
    Calculate accuracy metrics for CPU predictions.
    
    Args:
        actual_series: Series of actual CPU utilization values
        predicted_series: Series of predicted CPU utilization values
        
    Returns:
        Dictionary of accuracy metrics
    """
    # Make sure the series are aligned
    if len(actual_series) != len(predicted_series):
        return {"error": "Series lengths do not match"}
    
    if len(actual_series) == 0:
        return {"error": "Empty series provided"}
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(actual_series - predicted_series))
    
    # Calculate Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual_series - predicted_series) ** 2))
    
    # Calculate Mean Absolute Percentage Error
    # Avoid division by zero
    non_zeros = actual_series != 0
    if any(non_zeros):
        mape = np.mean(np.abs((actual_series[non_zeros] - predicted_series[non_zeros]) / actual_series[non_zeros])) * 100
    else:
        mape = float('nan')
    
    # Return all metrics
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

def aggregate_metrics(simulation_results: List[SimulationResult], 
                     all_metrics: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Aggregate metrics across multiple simulations into a summary DataFrame.
    
    Args:
        simulation_results: List of SimulationResult objects
        all_metrics: List of metrics dictionaries corresponding to the simulation results
        
    Returns:
        DataFrame with aggregated metrics
    """
    # Create a list to store the data for the DataFrame
    rows = []
    
    # Process each simulation result and its metrics
    for sim, metrics in zip(simulation_results, all_metrics):
        if not sim.scheduling_success:
            continue  # Skip failed simulations
            
        # Add a row with key information and metrics
        rows.append({
            'algorithm': sim.algorithm,
            'policy': sim.policy.name,
            'machine_id': sim.machine_id,
            'workload_type': sim.workload_type,
            'avg_cpu_actual': sim.avg_cpu_actual,
            'avg_cpu_predicted': sim.avg_cpu_predicted,
            'avg_solar_pv': sim.avg_solar_pv,
            'energy_consumed_wh': metrics.get('energy_consumed_wh'),
            'green_energy_wh': metrics.get('green_energy_wh'),
            'grid_energy_wh': metrics.get('grid_energy_wh'),
            'geuf_percentage': metrics.get('geuf_percentage')
        })
    
    # Create DataFrame from the rows
    if rows:
        results_df = pd.DataFrame(rows)
    else:
        # Create an empty DataFrame with the expected columns if no successful simulations
        results_df = pd.DataFrame(columns=[
            'algorithm', 'policy', 'machine_id', 'workload_type',
            'avg_cpu_actual', 'avg_cpu_predicted', 'avg_solar_pv',
            'energy_consumed_wh', 'green_energy_wh', 'grid_energy_wh', 'geuf_percentage'
        ])
    
    return results_df

def calculate_machine_averaged_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate machine-averaged metrics for each algorithm and policy combination.
    
    Args:
        results_df: DataFrame with aggregated metrics from aggregate_metrics()
        
    Returns:
        DataFrame with metrics averaged across machines for each algorithm and policy
    """
    if results_df.empty:
        return pd.DataFrame(columns=results_df.columns)
    
    # Group by algorithm and policy, then calculate means
    numeric_columns = [
        'avg_cpu_actual', 'avg_cpu_predicted', 'avg_solar_pv',
        'energy_consumed_wh', 'green_energy_wh', 'grid_energy_wh', 'geuf_percentage'
    ]
    
    # Select only the columns we need for grouping and averaging
    columns_to_use = ['algorithm', 'policy', 'workload_type'] + numeric_columns
    df_to_group = results_df[columns_to_use].copy()
    
    # Group by algorithm, policy, and workload_type and calculate mean
    averaged_df = df_to_group.groupby(['algorithm', 'policy', 'workload_type']).mean().reset_index()
    
    # Add a column to indicate this is machine-averaged data
    averaged_df['data_type'] = 'machine_averaged'
    
    return averaged_df
