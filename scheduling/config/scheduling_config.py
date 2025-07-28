"""
Configuration parameters for the workload scheduling simulation.
"""
from pathlib import Path

# Workload profile defines the characteristics of workloads to be scheduled
WORKLOAD_PROFILE = {
    'SMALL': {
        'cores_required': 2,  # Number of CPU cores required
        'duration_minutes': 60,  # Duration of the workload in minutes
        'min_cpu_utilization': 20,  # Minimum CPU utilization percentage
    },
    'MEDIUM': {
        'cores_required': 4,
        'duration_minutes': 120,
        'min_cpu_utilization': 30,
    },
    'LARGE': {
        'cores_required': 6,
        'duration_minutes': 180,
        'min_cpu_utilization': 40,
    },
}

# Server configuration
SERVER_CONFIG = {
    'total_cores_per_machine': 8,  # Total number of CPU cores per server
    'memory_gb': 32,  # Memory in GB (for potential future use)
}

# Power model for converting CPU utilization to power consumption
SERVER_POWER_MODEL = {
    'idle_power_watts': 100,  # Power consumption when server is idle
    'max_power_watts': 300,  # Maximum power consumption
    'power_per_core_watts': 25,  # Power consumption per active core
    'utilization_to_power_curve': 'linear',  # Relationship between CPU utilization and power
                                           # Options: 'linear', 'quadratic'
}

# Simulation parameters
SIMULATION_PARAMS = {
    'scheduling_window_hours': 24,  # How far ahead to look for scheduling opportunities
    'prediction_window_hours': 12,  # How far ahead predictions are considered reliable
    'time_step_minutes': 5,  # Resolution of the simulation (matching data sampling rate)
    'start_date': '2025-01-01',  # Arbitrary start date for simulation
    'algorithms': ['TempoSight'],  # Algorithms to include in the simulation
}

# Solar PV configuration
SOLAR_PV_CONFIG = {
    'data_path': Path(__file__).parent.parent / 'data' / 'data_obj.obj',
    'day_index': 0,  # Which day's data to use (0-indexed)
}

# Logging configuration
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"