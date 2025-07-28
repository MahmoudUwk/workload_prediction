"""
Script to generate visualizations from simulation results.
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import sys

# Add the project directory to the path so modules can be imported
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from src.visualization import plot_comparison_metric, plot_heatmap_comparison
    from config.scheduling_config import LOG_LEVEL
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure that the script is run from the 'scheduling' directory or that the PYTHONPATH is set correctly.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_visualizations(run_dir_path: Path, results_data: dict):
    """
    Generates and saves plots based on the aggregated simulation results.

    Args:
        run_dir_path (Path): Path to the specific run directory where plots will be saved.
        results_data (dict): A dictionary where keys are dataset names and values are pandas DataFrames
                             containing aggregated_metrics.
    """
    plots_dir = run_dir_path / "plots"
    plots_dir.mkdir(exist_ok=True) # Ensure plots directory exists

    logger.info(f"Generating visualizations in: {plots_dir}")

    for dataset_name, agg_df in results_data.items():
        if not isinstance(agg_df, pd.DataFrame) or agg_df.empty:
            logger.warning(f"Aggregated data for {dataset_name} is empty or not a DataFrame. Skipping plots.")
            continue

        logger.info(f"Plotting for dataset: {dataset_name}")

        # Plot GEUF comparison by algorithm
        geuf_file = plots_dir / f"{dataset_name}_geuf_by_algorithm.png"
        try:
            plot_comparison_metric(
                agg_df,
                "geuf_percentage",
                output_path=geuf_file,
                group_by='algorithm',
                title=f"Green Energy Utilization Factor by Algorithm - {dataset_name}"
            )
            logger.info(f"Saved GEUF comparison plot to {geuf_file}")
        except Exception as e:
            logger.error(f"Failed to generate GEUF comparison plot for {dataset_name}: {e}")

        # Plot energy consumption comparison by algorithm
        energy_file = plots_dir / f"{dataset_name}_energy_by_algorithm.png"
        try:
            plot_comparison_metric(
                agg_df,
                "energy_consumed_wh",
                output_path=energy_file,
                group_by='algorithm',
                title=f"Energy Consumption by Algorithm - {dataset_name}"
            )
            logger.info(f"Saved energy comparison plot to {energy_file}")
        except Exception as e:
            logger.error(f"Failed to generate energy comparison plot for {dataset_name}: {e}")

        # Plot heatmap of GEUF
        heatmap_file = plots_dir / f"{dataset_name}_geuf_heatmap.png"
        try:
            plot_heatmap_comparison(
                agg_df,
                value_col="geuf_percentage",
                output_path=heatmap_file,
                title=f"GEUF Heatmap - {dataset_name}"
            )
            logger.info(f"Saved GEUF heatmap to {heatmap_file}")
        except Exception as e:
            logger.warning(f"Could not generate GEUF heatmap for {dataset_name}: {e}")

def load_aggregated_results(run_data_dir: Path) -> dict:
    """
    Loads all machine-averaged CSV results from a given run data directory.
    Example filenames: Alibaba_machine_averaged_results.csv, BB_machine_averaged_results.csv
    """
    aggregated_results = {}
    logger.info(f"Loading aggregated results from: {run_data_dir}")
    for csv_file in run_data_dir.glob("*_machine_averaged_results.csv"):
        try:
            dataset_name = csv_file.name.replace("_machine_averaged_results.csv", "")
            df = pd.read_csv(csv_file)
            if not df.empty:
                aggregated_results[dataset_name] = df
                logger.info(f"Loaded {csv_file.name} for dataset {dataset_name}")
            else:
                logger.warning(f"File {csv_file.name} is empty.")
        except Exception as e:
            logger.error(f"Error loading {csv_file.name}: {e}")
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations from simulation results.")
    parser.add_argument(
        "run_directory",
        type=str,
        help="Path to the timestamped run directory (e.g., output/run_YYYYMMDD_HHMMSS)."
    )
    args = parser.parse_args()

    run_dir = Path(args.run_directory).resolve()
    run_data_dir = run_dir / "data"

    if not run_dir.is_dir() or not run_data_dir.is_dir():
        logger.error(f"Run directory '{run_dir}' or its 'data' subdirectory not found.")
        print(f"Error: Run directory '{run_dir}' or its 'data' subdirectory not found.")
        print("Please provide a valid path to a run directory created by main.py.")
        sys.exit(1)

    logger.info(f"Processing results from run directory: {run_dir}")
    
    # Load the aggregated data
    all_agg_results = load_aggregated_results(run_data_dir)

    if not all_agg_results:
        logger.warning("No aggregated results found to generate plots.")
        print("No aggregated result files (*_machine_averaged_results.csv) found in the specified run's data directory.")
        return

    # Generate visualizations
    create_visualizations(run_dir, all_agg_results)

    logger.info("Visualization generation complete.")

if __name__ == "__main__":
    main()
