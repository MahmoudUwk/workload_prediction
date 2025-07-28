"""
Main script to run simulations and generate outputs.

This script orchestrates the entire simulation process:
1. Loading configuration
2. Loading and aligning datasets
3. Running simulations for different policies, algorithms, and machines
4. Calculating metrics
5. Generating visualizations and summary tables
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime

# Add the project directory to the path so modules can be imported
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from config.scheduling_config import (
    WORKLOAD_PROFILE, SERVER_CONFIG, SERVER_POWER_MODEL, 
    SIMULATION_PARAMS, SOLAR_PV_CONFIG, LOG_LEVEL
)
from load_CPU_data import get_aligned_datasets
from src.simulation_engine import simulate, SchedulingPolicy
from src.metrics_calculator import (
    calculate_simulation_metrics, calculate_prediction_accuracy, 
    aggregate_metrics, calculate_machine_averaged_metrics
)
from src.visualization import (
    plot_comparison_metric, plot_individual_schedule_details,
    plot_heatmap_comparison
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_output_directories():
    """Create output directories for results if they don't exist."""
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different types of outputs
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    
    # Create a timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    
    return run_dir

def run_simulations(dataset_name, aligned_data, workload_type="MEDIUM"):
    """
    Run simulations for all algorithms and machines in the dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('Alibaba' or 'BB')
        aligned_data (dict): The aligned dataset from get_aligned_datasets
        workload_type (str): Type of workload to simulate (from WORKLOAD_PROFILE)
        
    Returns:
        tuple: (simulation_results, metrics_results) lists of results
    """
    logger.info(f"Starting simulations for dataset: {dataset_name}, workload: {workload_type}")
    
    if dataset_name not in aligned_data:
        logger.error(f"Dataset '{dataset_name}' not found in aligned data")
        return [], []
    
    if workload_type not in WORKLOAD_PROFILE:
        logger.error(f"Workload type '{workload_type}' not defined in WORKLOAD_PROFILE")
        return [], []
    
    workload_cfg = WORKLOAD_PROFILE[workload_type]
    all_simulation_results = []
    all_metrics_results = []
    
    # Get the dataset-specific aligned data
    dataset_data = aligned_data[dataset_name]
    
    # Track the number of simulations
    total_simulations = 0
    successful_simulations = 0
    
    # For each algorithm
    for algo_name, machines_data in dataset_data.items():
        logger.info(f"Processing algorithm: {algo_name}")
        
        # For each machine
        for machine_id, machine_df in machines_data.items():
            logger.debug(f"Processing machine: {machine_id}")
            
            # Run simulations with each policy
            for policy in SchedulingPolicy:
                total_simulations += 1
                
                # Run the simulation
                try:
                    sim_result = simulate(
                        machine_df=machine_df,
                        machine_id=machine_id,
                        algorithm=algo_name,
                        workload_cfg=workload_cfg,
                        server_cfg=SERVER_CONFIG,
                        power_model=SERVER_POWER_MODEL,
                        policy=policy,
                        workload_type=workload_type
                    )
                    
                    all_simulation_results.append(sim_result)
                    
                    # Calculate metrics if the simulation was successful
                    if sim_result.scheduling_success:
                        successful_simulations += 1
                        metrics = calculate_simulation_metrics(
                            simulation_result=sim_result,
                            machine_df=machine_df,
                            workload_cfg=workload_cfg,
                            power_model=SERVER_POWER_MODEL
                        )
                    else:
                        logger.debug(f"Simulation failed for {algo_name}, {machine_id}, {policy}: {sim_result.reason_if_failed}")
                        metrics = {
                            "scheduling_success": False,
                            "failure_reason": sim_result.reason_if_failed
                        }
                    
                    all_metrics_results.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error in simulation for {algo_name}, {machine_id}, {policy}: {e}")
                    # Add a placeholder for failed simulations
                    all_simulation_results.append(None)
                    all_metrics_results.append({"scheduling_success": False, "failure_reason": str(e)})
    
    logger.info(f"Completed simulations for dataset {dataset_name}: "
                f"{successful_simulations} successful out of {total_simulations} total")
    
    return all_simulation_results, all_metrics_results

def main():
    """Main function to run the entire simulation workflow."""
    start_time = time.time()
    logger.info("Starting carbon-intelligent workload scheduling simulation")
    
    # Setup output directories
    run_dir = setup_output_directories()
    logger.info(f"Results will be saved to: {run_dir}")
    
    # Load and align data
    logger.info("Loading and aligning datasets...")
      # Create config for data alignment
    data_config = {
        'start_time': pd.Timestamp(SIMULATION_PARAMS['start_date']),
        'day_index': SOLAR_PV_CONFIG.get('day_index', 0),
        'algorithms': SIMULATION_PARAMS.get('algorithms', None)  # Selected algorithms to process
    }
    
    try:
        # Load data for both datasets
        aligned_data = {}
        
        # Try loading Alibaba dataset
        try:
            logger.info("Loading Alibaba dataset...")
            alibaba_data = get_aligned_datasets('Alibaba', SOLAR_PV_CONFIG['data_path'], data_config)
            aligned_data.update(alibaba_data)
            logger.info(f"Alibaba dataset loaded with {sum(len(algo) for algo in alibaba_data.get('Alibaba', {}).values())} machine entries")
        except Exception as e:
            logger.warning(f"Error loading Alibaba dataset: {e}")
        
        # Try loading BB dataset
        try:
            logger.info("Loading BB (BitBrains) dataset...")
            bb_data = get_aligned_datasets('BB', SOLAR_PV_CONFIG['data_path'], data_config)
            aligned_data.update(bb_data)
            logger.info(f"BB dataset loaded with {sum(len(algo) for algo in bb_data.get('BB', {}).values())} machine entries")
        except Exception as e:
            logger.warning(f"Error loading BB dataset: {e}")
            
        if not aligned_data:
            logger.error("No datasets could be loaded. Exiting.")
            return
            
        # Define workload types to iterate over
        workload_types_to_run = ['SMALL', 'MEDIUM', 'LARGE']
        
        for workload_type in workload_types_to_run:
            logger.info(f"Processing simulations for WORKLOAD TYPE: {workload_type}")
            
            # Run simulations for each dataset for the current workload_type
            all_results_current_workload = {}
            all_metrics_current_workload = {}
            
            for dataset_name in aligned_data.keys():
                logger.info(f"Running simulations for {dataset_name} dataset, workload: {workload_type}...")
                results, metrics = run_simulations(dataset_name, aligned_data, workload_type)
                all_results_current_workload[dataset_name] = results
                all_metrics_current_workload[dataset_name] = metrics
                
            # Aggregate results for the current workload_type
            logger.info(f"Aggregating results for WORKLOAD TYPE: {workload_type}...")
            
            for dataset_name, results in all_results_current_workload.items():
                valid_results = [r for r in results if r is not None]
                valid_metrics = [m for m in all_metrics_current_workload[dataset_name] if m is not None]
                
                if valid_results and valid_metrics:
                    agg_df = aggregate_metrics(valid_results, valid_metrics)
                    
                    # Save aggregated results with workload_type in filename
                    results_file = run_dir / "data" / f"{dataset_name}_{workload_type}_results.csv"
                    agg_df.to_csv(results_file)
                    logger.info(f"Results for {workload_type} saved to {results_file}")
                    
                    # Calculate and save machine-averaged metrics
                    logger.info(f"Calculating machine-averaged metrics for {workload_type}...")
                    avg_df = calculate_machine_averaged_metrics(agg_df)
                    avg_results_file = run_dir / "data" / f"{dataset_name}_{workload_type}_machine_averaged_results.csv"
                    avg_df.to_csv(avg_results_file)
                    logger.info(f"Machine-averaged results for {workload_type} saved to {avg_results_file}")
                else:
                    logger.warning(f"No valid results for {dataset_name}, workload: {workload_type}")
        
        logger.info(f"All simulations completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to {run_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred during simulation execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
