"""
Simulation engine for workload scheduling based on different policies.
"""
import pandas as pd
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

class SchedulingPolicy(Enum):
    """Defines different scheduling policies for comparison."""
    PREDICTIVE_CARBON_AWARE = auto()  # Uses predicted CPU and solar PV data
    NAIVE_CARBON_AWARE = auto()       # Uses actual CPU and solar PV data (oracle/best case)
    GREEN_OBLIVIOUS = auto()          # Uses only CPU data, ignoring solar availability

@dataclass
class SimulationResult:
    """Stores the results of a scheduling simulation."""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    machine_id: str
    algorithm: str
    policy: SchedulingPolicy
    avg_cpu_actual: float
    avg_cpu_predicted: float
    avg_solar_pv: float
    workload_type: str
    max_cpu_during_workload: float
    min_cpu_during_workload: float
    scheduling_success: bool
    reason_if_failed: str = None

def calculate_energy_score(cpu_data: pd.Series, solar_data: pd.Series, 
                          power_model: Dict[str, Any]) -> float:
    """
    Calculate a scheduling score based on CPU utilization and solar availability.
    Higher score is better (more solar availability, lower estimated power consumption).
    
    Args:
        cpu_data: CPU utilization series (percentage)
        solar_data: Solar PV power series (arbitrary units, higher is better)
        power_model: Power model parameters
        
    Returns:
        float: Energy score, higher is better for scheduling
    """
    # Normalize CPU to 0-1 scale for power calculation
    normalized_cpu_for_power_calc = cpu_data / 100.0
    
    # Calculate estimated power draw based on CPU utilization
    idle_power = power_model.get('idle_power_watts', 50) 
    max_power = power_model.get('max_power_watts', 200)  
    
    if power_model.get('utilization_to_power_curve') == 'quadratic':
        estimated_power_draw = idle_power + (max_power - idle_power) * (normalized_cpu_for_power_calc ** 2)
    else: # linear
        estimated_power_draw = idle_power + (max_power - idle_power) * normalized_cpu_for_power_calc
    
    # Normalize estimated_power_draw to a 0-1 scale (lower is better)
    # where 0 corresponds to idle_power and 1 corresponds to max_power.
    if max_power > idle_power:
        normalized_estimated_power = (estimated_power_draw - idle_power) / (max_power - idle_power)
    elif max_power == idle_power: # If max and idle are same, power is constant
        normalized_estimated_power = pd.Series(0.0, index=cpu_data.index) # No variable component
    else: # Should not happen, but as a fallback, treat as max consumption
        normalized_estimated_power = pd.Series(1.0, index=cpu_data.index)

    # Ensure normalized_estimated_power is clipped between 0 and 1, as CPU util might be outside [0,100]
    # or power model might be misconfigured.
    normalized_estimated_power = normalized_estimated_power.clip(0, 1)

    # Normalize solar data to 0-1 scale (higher is better)
    # Ensure solar_data is a Series for .max() to work as expected, or handle scalar if possible
    if isinstance(solar_data, pd.Series) and not solar_data.empty:
        max_solar = solar_data.max()
    else: # Fallback for empty series or scalar solar_data, though solar_data should be a series
        max_solar = 0.0 # Or handle as an error/warning

    max_solar = max(max_solar, 1.0) # Avoid divide by zero if max_solar is 0
    normalized_solar = solar_data / max_solar
    normalized_solar = normalized_solar.clip(0,1) # Ensure solar normalization is also clipped

    # Calculate score: Higher solar * (1 - lower normalized_estimated_power) = better score
    # This score ranges from 0 to 1.
    # It rewards high solar and low power consumption simultaneously.
    score = normalized_solar * (1 - normalized_estimated_power)
    
    return score.mean()

def find_best_scheduling_window(machine_df: pd.DataFrame, 
                              workload_cfg: Dict[str, Any],
                              policy: SchedulingPolicy,
                              power_model: Dict[str, Any]) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], bool, Optional[str]]:
    """
    Find the best time window to schedule a workload based on the specified policy.
    
    Args:
        machine_df: DataFrame with cpu_actual, cpu_predicted, solar_pv_aligned columns
        workload_cfg: Workload configuration (duration, minimum requirements)
        policy: Scheduling policy to use
        power_model: Power model parameters
    
    Returns:
        tuple: (best_start_time, best_end_time, schedule_successful, reason_if_failed)
    """
    # Get workload duration in minutes
    duration_minutes = workload_cfg['duration_minutes']
    
    # Minimum CPU utilization threshold
    min_cpu_threshold = workload_cfg.get('min_cpu_utilization', 0)
    
    # Create a copy of the dataframe with only the needed columns based on policy
    if policy == SchedulingPolicy.PREDICTIVE_CARBON_AWARE:
        cpu_column = 'cpu_predicted'
        df_copy = machine_df[['cpu_predicted', 'solar_pv_aligned']].copy()
        
    elif policy == SchedulingPolicy.NAIVE_CARBON_AWARE:
        cpu_column = 'cpu_actual'
        df_copy = machine_df[['cpu_actual', 'solar_pv_aligned']].copy()
        
    elif policy == SchedulingPolicy.GREEN_OBLIVIOUS:
        cpu_column = 'cpu_predicted'
        df_copy = machine_df[[cpu_column]].copy()
        df_copy['solar_pv_aligned'] = 1.0
    
    # Calculate window sizes
    window_size = pd.Timedelta(minutes=duration_minutes)
    
    # Check if we have enough data points
    if len(df_copy) < 2:
        return None, None, False, "Not enough data points in machine_df"
    
    # Calculate time step (difference between consecutive timestamps)
    time_step = df_copy.index[1] - df_copy.index[0]
    
    # Calculate number of time steps in the window
    steps_in_window = int(window_size / time_step)
    
    if steps_in_window <= 0:
        return None, None, False, "Workload duration is too short for data resolution"
    
    # Prepare to store scores for each possible window
    window_scores = []

    # Iterate over all possible start times for the window
    for i in range(len(df_copy) - steps_in_window + 1):
        start_time = df_copy.index[i]
        end_time = start_time + window_size - time_step
        
        current_window_df = df_copy.loc[start_time:end_time]

        if len(current_window_df) < steps_in_window:
            continue

        current_cpu_window = current_window_df[cpu_column]
        current_solar_window = current_window_df['solar_pv_aligned']
        
        # Check if CPU utilization is sufficient
        if current_cpu_window.mean() < min_cpu_threshold:
            continue

        # Calculate score for this window
        score = calculate_energy_score(current_cpu_window, current_solar_window, power_model)
        window_scores.append({'start_time': start_time, 'score': score})

    if not window_scores:
        return None, None, False, "No suitable scheduling window found"

    # Select the window with the highest score
    best_window = max(window_scores, key=lambda x: x['score'])
    best_start_time = best_window['start_time']
    best_end_time = best_start_time + window_size - time_step

    return best_start_time, best_end_time, True, None

def simulate(machine_df: pd.DataFrame, 
             machine_id: str,
             algorithm: str,  
             workload_cfg: Dict[str, Any],
             server_cfg: Dict[str, Any],
             power_model: Dict[str, Any],
             policy: SchedulingPolicy,
             workload_type: str) -> Optional[SimulationResult]:
    """
    Simulate workload scheduling based on the specified policy.
    
    Args:
        machine_df: DataFrame with cpu_actual, cpu_predicted, solar_pv_aligned data
        machine_id: ID of the machine being simulated
        algorithm: Name of the prediction algorithm
        workload_cfg: Workload configuration
        server_cfg: Server configuration
        power_model: Power consumption model
        policy: Scheduling policy to use
        workload_type: Type of workload (e.g., 'SMALL', 'MEDIUM', 'LARGE')
        
    Returns:
        SimulationResult object containing scheduling details
    """
    # Find the best scheduling window using the chosen policy
    machine_df_for_policy = machine_df.copy()

    best_start_time, best_end_time, schedule_success, reason = find_best_scheduling_window(
        machine_df_for_policy,
        workload_cfg,
        policy,
        power_model
    )

    if not schedule_success:
        return SimulationResult(
            start_time=None,
            end_time=None,
            machine_id=machine_id,
            algorithm=algorithm,
            policy=policy,
            avg_cpu_actual=None,
            avg_cpu_predicted=None,
            avg_solar_pv=None,
            workload_type=workload_type,
            max_cpu_during_workload=None,
            min_cpu_during_workload=None,
            scheduling_success=False,
            reason_if_failed=reason
        )
    
    # Get the actual data for the selected window
    window_data = machine_df.loc[best_start_time:best_end_time]
    
    # Calculate average metrics for the scheduled window
    avg_cpu_actual = window_data['cpu_actual'].mean()
    avg_cpu_predicted = window_data['cpu_predicted'].mean()
    avg_solar_pv = window_data['solar_pv_aligned'].mean()
    max_cpu = window_data['cpu_actual'].max()
    min_cpu = window_data['cpu_actual'].min()
    
    # Create and return the simulation result
    result = SimulationResult(
        start_time=best_start_time,
        end_time=best_end_time,
        machine_id=machine_id,
        algorithm=algorithm,
        policy=policy,
        avg_cpu_actual=avg_cpu_actual,
        avg_cpu_predicted=avg_cpu_predicted,
        avg_solar_pv=avg_solar_pv,
        workload_type=workload_type,
        max_cpu_during_workload=max_cpu,
        min_cpu_during_workload=min_cpu,
        scheduling_success=True,
        reason_if_failed=None
    )
    
    return result
