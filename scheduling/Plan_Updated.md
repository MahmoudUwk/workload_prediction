# Carbon-Intelligent Workload-Scheduling: Updated Implementation Plan

## 1. Project Goal
Demonstrate, with rigorous experiments, that **external CPU-utilisation prediction algorithms** (TempoSight, LSTM, PatchTST …) enable _measurably greener_ workload scheduling compared with (a) naive forecasts and (b) green-oblivious baselines.

## 2. Core Idea & Data Handling Strategy

The project will use CPU utilization predictions to make energy-aware scheduling decisions.

*   **Data Source Script**: The existing `load_CPU_data.py` script has been enhanced and serves as the primary tool for loading and preparing all necessary data.
*   **CPU Data**:
    *   Actual (`y_test`) and predicted (`y_pred`) CPU utilization, along with machine IDs (`Mids_test`).
    *   Sourced from `.obj` files for 'Alibaba' and 'BB' datasets.
    *   Sampling rate: **5 minutes**.
    *   CPU utilization is processed as percentages (0-100%). Diagnostic logging has been added to `load_CPU_data.py` to verify the min/max ranges of loaded CPU data.
*   **Solar PV Data**:
    *   Solar power generation (`pv_power`) from `scheduling/data/data_obj.obj`.
    *   Original sampling rate: **15 minutes**.
*   **Temporal Alignment and Normalization**:
    *   The `load_CPU_data.py` script is responsible for upsampling the solar PV data to a **5-minute** resolution using linear interpolation.
    *   The upsampled solar PV data for the entire dataset/day is then **normalized to a [0,1] range**.
    *   This normalized 5-minute solar data is aligned with each machine's 5-minute CPU data, ensuring a common `DatetimeIndex` and a consistent scale for solar availability.
*   **Algorithm Selection**:
    *   The implementation now supports filtering to specific algorithms (e.g., **TempoSight** and **CEDL**) for focused analysis.
    *   This is configurable via the `algorithms` parameter in `SIMULATION_PARAMS`.
    *   Users can specify a list of desired algorithms in the configuration, and only those will be loaded and processed.
    *   Available algorithms include: TempoSight, CEDL, LSTM, PatchTST, CNN, Adaptive, GBT, SVR, and LR.
    *   If no algorithm list is specified, all available algorithms will be used.
*   **Machine-Averaged Results**:
    *   Results are now available both at the individual machine level and as machine-averaged metrics.
    *   Averaging provides a clearer picture of algorithm performance across different machines.
    *   Machine-averaged metrics are saved to separate CSV files for analysis.
*   **Prepared Data Structure**: The data preparation process yields a structure like:
    `{dataset_name: {algorithm_name: {machine_id: pd.DataFrame}}}`
    Where each `DataFrame` contains columns: `timestamp (index)`, `cpu_actual`, `cpu_predicted`, `solar_pv_aligned` (normalized to [0,1]).

## 3. Project Layout

```
scheduling/
├── load_CPU_data.py            # Enhanced script for all data loading, prep & alignment
├── config/
│   └── scheduling_config.py    # Simulation parameters, workload/server profiles
├── src/
│   ├── simulation_engine.py    # Core scheduling logic and policies
│   ├── metrics_calculator.py   # Calculates GEUF, energy, prediction errors, machine averages
│   └── visualization.py        # Plotting results (comparisons)
├── main.py                     # Main script to run simulations and generate outputs
├── output/                     # Directory for simulation results
│   ├── data/                   # CSV files with metrics
│   └── plots/                  # Generated visualizations
├── run_TIMESTAMP/              # Results from specific simulation runs 
│   ├── data/                   # CSV files for each dataset
│   │   ├── Dataset_results.csv                # Per-machine results
│   │   └── Dataset_machine_averaged_results.csv # Aggregated results
│   └── plots/                  # Generated visualizations
└── test/
    ├── test_simulation_engine.py # Unit tests for scheduling logic
    ├── test_metrics_calculator.py # Unit tests for metrics
    └── test_algo_selection.py    # Script to test algorithm selection functionality
```

## 4. Implementation Details

### Step 1: Configuration (`config/scheduling_config.py`)
*   **Objective**: Define parameters for the simulation environment.
*   **Details**:
    *   `WORKLOAD_PROFILE`: Define characteristics of workloads to be scheduled (e.g., CPU cores required, duration).
    *   `SERVER_CONFIG`: Define server properties (e.g., total cores per machine, e.g., 8).
    *   `SERVER_POWER_MODEL`: Simple model for power consumption (e.g., idle power, max power, power per core).
    *   `SIMULATION_PARAMS`: 
        * Scheduling window to consider
        * `algorithms`: List of prediction algorithms to include (e.g., ['TempoSight', 'CEDL'])
        * This parameter is used to filter which algorithms are processed during data loading
    *   `LOG_LEVEL`: For controlling script output ("INFO" or "DEBUG").

### Step 2: Enhanced Data Preparation (`load_CPU_data.py`)
*   **Objective**: Load and prepare data for the simulation, ensuring consistent scaling.
*   **Implementation**:
    *   Added `get_aligned_datasets(dataset_name, solar_obj_path, config)` function that:
        * Loads CPU utilization data from specified datasets.
        * Filters to selected algorithms if specified in config.
        * Processes multi-dimensional arrays for machine-specific data.
        * Upsamples solar PV data to 5-minute resolution.
        * **Normalizes the entire day's upsampled solar PV data to a [0,1] range.** This `solar_pv_aligned` data is then merged with CPU data.
        * Adds diagnostic logging for the min/max range of `cpu_actual` and `cpu_predicted` for the first processed machine per dataset to help verify expected scaling (e.g., 0-100).
        * Returns the nested dictionary structure.
    *   Algorithm filtering occurs in the loop that processes each algorithm:
        ```python
        # Process each algorithm's data
        for algo_name, data_dict in algo_data_map.items():
            # Skip if algorithm not in selected algorithms list (if provided in config)
            selected_algos = config.get('algorithms', None)
            if selected_algos and algo_name not in selected_algos:
                print(f"Skipping {algo_name} as it's not in the selected algorithms list.")
                continue
                
            # Process the selected algorithm...
        ```

### Step 3: Simulation Engine (`src/simulation_engine.py`)
*   **Objective**: Implement the core logic for scheduling workloads based on different policies.
*   **Components**:
    *   `SchedulingPolicy` (Enum): PREDICTIVE_CARBON_AWARE, NAIVE_CARBON_AWARE, GREEN_OBLIVIOUS
    *   `SimulationResult` (dataclass): Stores scheduling decisions and metrics
    *   `calculate_energy_score()`: Evaluates potential scheduling windows.
        *   The `power_model` is now correctly passed and utilized.
        *   The score is calculated based on both normalized solar availability [0,1] and normalized estimated power consumption [0,1] (derived from the `power_model`).
        *   The scoring formula is: `score = normalized_solar * (1 - normalized_estimated_power)`, aiming for a [0,1] range where higher is better.
    *   `find_best_scheduling_window()`: Identifies optimal time slots based on the energy score.
    *   `simulate()`: Main function for running scheduling simulations.

### Step 4: Metrics Calculator (`src/metrics_calculator.py`)
*   **Objective**: Quantify the performance of different scheduling policies.
*   **Key Functions**:
    *   `calculate_simulation_metrics()`: Computes energy consumption and green energy utilization
    *   `calculate_prediction_accuracy()`: Evaluates CPU prediction performance
    *   `aggregate_metrics()`: Combines and summarizes results across simulations
    *   `calculate_machine_averaged_metrics()`: Averages results across machines for each algorithm-policy pair
        * Groups results by algorithm, policy, and workload type
        * Calculates mean values for all numeric metrics
        * Facilitates easier comparison of algorithm performance

### Step 5: Visualization (`src/visualization.py`)
*   **Objective**: Create visual summaries of the simulation results.
*   **Key Functions**:
    *   `plot_comparison_metric()`: Bar charts comparing algorithms/policies
        * Already incorporates machine averaging via grouping
        * Generates plots showing average metrics by algorithm and policy
    *   `plot_individual_schedule_details()`: Detailed view of specific scheduling decisions
    *   `plot_heatmap_comparison()`: Heatmap visualization of metrics
    *   All visualization functions automatically perform the necessary grouping and averaging

### Step 6: Main Script (`main.py`)
*   **Objective**: Orchestrate the entire simulation workflow.
*   **Workflow**:
    1.  Load configurations from `scheduling_config.py`
    2.  Pass selected algorithms to `get_aligned_datasets()` via the data_config dictionary
    3.  Run simulations for each dataset, algorithm, and machine
    4.  Calculate and aggregate metrics for each machine
    5.  Calculate machine-averaged metrics for clearer comparison
    6.  Save both per-machine and machine-averaged results to CSV files
    7.  Generate visualizations and save results

## 5. Testing Focus
*   Unit tests for both the simulation engine and metrics calculator
*   Verify handling of multi-dimensional data arrays
*   Ensure proper algorithm filtering
*   The `test_algo_selection.py` script provides a simple test to verify that:
    * Only specified algorithms are loaded (e.g., TempoSight and CEDL)
    * Other algorithms are properly skipped during data loading
    * The algorithm selection feature works as expected
*   Validate machine-averaged metrics calculations

## 6. Current Implementation Status
*   Data loading and alignment: **Completed**
*   Algorithm selection feature: **Implemented**
    * Filtering functionality added to `load_CPU_data.py`
    * Configuration parameter added to `scheduling_config.py`
    * Verified with `test_algo_selection.py`
*   Core simulation engine: **Completed**
*   Metrics calculation: **Completed**
    * Individual machine metrics: **Completed**
    * Machine-averaged metrics: **Implemented**
*   Result visualization: **Completed**
    * Bar charts and heatmaps using machine-averaged data for clarity
*   Focus on TempoSight and CEDL algorithms for initial analysis

## 7. Expected Outcomes
*   Quantitative comparison of different prediction algorithms' impact on green energy utilization
*   Insights into the effectiveness of predictive scheduling versus naive approaches
*   Clear visualizations demonstrating the benefits of carbon-aware scheduling
*   Machine-averaged metrics for more representative algorithm comparisons

## 8. How to Use Algorithm Selection

To specify which algorithms to use in simulations:

1. Open `config/scheduling_config.py`
2. Modify the `SIMULATION_PARAMS` dictionary:
   ```python
   SIMULATION_PARAMS = {
       'scheduling_window_hours': 24,
       'prediction_window_hours': 12,
       'time_step_minutes': 5,
       'start_date': '2025-01-01',
       'algorithms': ['TempoSight', 'CEDL'],  # Specify your desired algorithms here
   }
   ```
3. Available algorithm options:
   * 'TempoSight' - The TempoSight algorithm
   * 'CEDL' - CEDL algorithm
   * 'LSTM' - LSTM algorithm
   * 'Patch\nTST' - PatchTST algorithm
   * 'CNN' - CNN algorithm
   * 'Adaptive' - Adaptive predictor
   * 'GBT' - Gradient Boosting Trees
   * 'SVR' - Support Vector Regression
   * 'LR' - Linear Regression

4. When running the simulation via `main.py`, only the specified algorithms will be processed, saving computation time and allowing focused analysis.

## 9. Output Files and Visualizations

After running simulations, two types of result files are generated:

1. **Per-machine results** (`Dataset_results.csv`):
   * Contains detailed metrics for each machine, algorithm, and policy combination
   * Useful for in-depth analysis and understanding variance across machines

2. **Machine-averaged results** (`Dataset_machine_averaged_results.csv`):
   * Contains metrics averaged across all machines for each algorithm-policy pair
   * Provides a clearer picture of overall algorithm performance
   * Used as the basis for visualization plots

3. **Visualizations**:
   * All plots (bar charts, heatmaps) are generated using the machine-averaged data
   * This ensures that visualizations represent general algorithm performance rather than being skewed by outlier machines
   * Comparisons show performance differences between algorithms and policies based on averaged metrics
