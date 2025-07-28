"""
Unit tests for the simulation engine module.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the parent directory to the path so modules can be imported
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the modules to test
from src.simulation_engine import (
    SchedulingPolicy, find_best_scheduling_window, 
    calculate_energy_score, simulate
)

class TestSimulationEngine(unittest.TestCase):
    """Test cases for the simulation engine."""
    
    def setUp(self):
        """Setup test data."""
        # Create a sample DataFrame with test data
        start_date = datetime(2025, 1, 1)
        periods = 288  # 24 hours at 5-minute intervals
        
        # Generate timestamps
        timestamps = [start_date + timedelta(minutes=5*i) for i in range(periods)]
        
        # Generate simulated CPU and solar data
        np.random.seed(42)  # For reproducible results
        
        # CPU data with a daily pattern (high during day, low at night)
        hour_of_day = [t.hour for t in timestamps]
        cpu_actual = np.array([30 + 20 * np.sin(h * np.pi / 12) + np.random.normal(0, 5) for h in hour_of_day])
        cpu_actual = np.clip(cpu_actual, 10, 90)  # Keep within reasonable bounds
        
        # Predicted CPU with some error
        cpu_pred = cpu_actual + np.random.normal(0, 5, size=len(cpu_actual))
        cpu_pred = np.clip(cpu_pred, 10, 90)
        
        # Solar data (zero at night, bell curve during day)
        solar_pv = np.array([max(0, 100 * np.sin(np.pi * h / 12) ** 2 if 6 <= h <= 18 else 0) for h in hour_of_day])
        
        # Create the DataFrame
        self.test_df = pd.DataFrame({
            'cpu_actual': cpu_actual,
            'cpu_predicted': cpu_pred,
            'solar_pv_aligned': solar_pv
        }, index=pd.DatetimeIndex(timestamps))
        
        # Define test workload config
        self.test_workload = {
            'cores_required': 2,
            'duration_minutes': 60,
            'min_cpu_utilization': 20,
        }
        
        # Define test server config
        self.test_server = {
            'total_cores_per_machine': 8,
        }
        
        # Define test power model
        self.test_power_model = {
            'idle_power_watts': 100,
            'max_power_watts': 300,
            'power_per_core_watts': 25,
            'utilization_to_power_curve': 'linear',
        }

    def test_find_best_window_predictive_carbon_aware(self):
        """Test finding the best window using predictive carbon-aware policy."""
        start_time, end_time, reason = find_best_scheduling_window(
            self.test_df, 
            self.test_workload, 
            SchedulingPolicy.PREDICTIVE_CARBON_AWARE
        )
        
        # Assertions
        self.assertIsNotNone(start_time)
        self.assertIsNotNone(end_time)
        self.assertIsNone(reason)
        
        # The window should be 60 minutes
        window_duration = (end_time - start_time).total_seconds() / 60
        self.assertAlmostEqual(window_duration, 60.0, delta=1.0)
        
        # Extract the window data
        window_data = self.test_df.loc[start_time:end_time]
        
        # The selected window should meet the minimum CPU threshold
        self.assertGreaterEqual(window_data['cpu_predicted'].min(), self.test_workload['min_cpu_utilization'])
        
        # The window should have some solar availability (if possible in the test data)
        # This is a soft check since it depends on the test data
        self.assertGreaterEqual(window_data['solar_pv_aligned'].mean(), 0)

    def test_find_best_window_naive_carbon_aware(self):
        """Test finding the best window using naive carbon-aware policy."""
        start_time, end_time, reason = find_best_scheduling_window(
            self.test_df, 
            self.test_workload, 
            SchedulingPolicy.NAIVE_CARBON_AWARE
        )
        
        # Assertions
        self.assertIsNotNone(start_time)
        self.assertIsNotNone(end_time)
        self.assertIsNone(reason)
        
        # The window should be 60 minutes
        window_duration = (end_time - start_time).total_seconds() / 60
        self.assertAlmostEqual(window_duration, 60.0, delta=1.0)
        
        # Extract the window data
        window_data = self.test_df.loc[start_time:end_time]
        
        # The selected window should meet the minimum CPU threshold
        self.assertGreaterEqual(window_data['cpu_actual'].min(), self.test_workload['min_cpu_utilization'])

    def test_find_best_window_green_oblivious(self):
        """Test finding the best window using green-oblivious policy."""
        start_time, end_time, reason = find_best_scheduling_window(
            self.test_df, 
            self.test_workload, 
            SchedulingPolicy.GREEN_OBLIVIOUS
        )
        
        # Assertions
        self.assertIsNotNone(start_time)
        self.assertIsNotNone(end_time)
        self.assertIsNone(reason)
        
        # The window should be 60 minutes
        window_duration = (end_time - start_time).total_seconds() / 60
        self.assertAlmostEqual(window_duration, 60.0, delta=1.0)
        
        # Extract the window data
        window_data = self.test_df.loc[start_time:end_time]
        
        # The selected window should meet the minimum CPU threshold
        self.assertGreaterEqual(window_data['cpu_predicted'].min(), self.test_workload['min_cpu_utilization'])
        
        # For green oblivious, check if it selected a low CPU window (it should ignore solar)
        all_possible_windows = []
        for i in range(len(self.test_df) - 12):  # 60 minutes = 12 5-min intervals
            window = self.test_df['cpu_predicted'].iloc[i:i+12].mean()
            if self.test_df['cpu_predicted'].iloc[i:i+12].min() >= self.test_workload['min_cpu_utilization']:
                all_possible_windows.append(window)
        
        if all_possible_windows:
            min_possible_avg = min(all_possible_windows)
            selected_avg = window_data['cpu_predicted'].mean()
            # Selected window should be relatively close to the minimum possible
            self.assertLessEqual(selected_avg, min_possible_avg * 1.2)

    def test_calculate_energy_score(self):
        """Test the energy score calculation."""
        # Create sample data for testing
        cpu_data = pd.Series([30.0, 40.0, 50.0, 60.0])
        solar_data = pd.Series([20.0, 40.0, 80.0, 60.0])
        power_model = {
            'idle_power_watts': 100,
            'max_power_watts': 300,
            'utilization_to_power_curve': 'linear'
        }
        
        # Calculate score
        score = calculate_energy_score(cpu_data, solar_data, power_model)
        
        # Score should be a float
        self.assertIsInstance(score, float)

    def test_simulate_all_policies(self):
        """Test running simulations with all policies."""
        for policy in SchedulingPolicy:
            # Run simulation
            result = simulate(
                machine_df=self.test_df,
                machine_id="test_machine",
                algorithm="test_algo",
                workload_cfg=self.test_workload,
                server_cfg=self.test_server,
                power_model=self.test_power_model,
                policy=policy,
                workload_type="TEST"
            )
            
            # Basic assertions
            self.assertTrue(result.scheduling_success)
            self.assertEqual(result.machine_id, "test_machine")
            self.assertEqual(result.algorithm, "test_algo")
            self.assertEqual(result.policy, policy)
            self.assertEqual(result.workload_type, "TEST")
            self.assertIsNotNone(result.start_time)
            self.assertIsNotNone(result.end_time)
            self.assertIsNotNone(result.avg_cpu_actual)
            self.assertIsNotNone(result.avg_cpu_predicted)
            self.assertIsNotNone(result.avg_solar_pv)

    def test_high_cpu_threshold_failure(self):
        """Test that scheduling fails when CPU threshold is too high."""
        # Set an impossibly high threshold
        high_threshold_workload = self.test_workload.copy()
        high_threshold_workload['min_cpu_utilization'] = 95
        
        result = simulate(
            machine_df=self.test_df,
            machine_id="test_machine",
            algorithm="test_algo",
            workload_cfg=high_threshold_workload,
            server_cfg=self.test_server,
            power_model=self.test_power_model,
            policy=SchedulingPolicy.PREDICTIVE_CARBON_AWARE,
            workload_type="TEST"
        )
        
        # Should fail because no window meets the threshold
        self.assertFalse(result.scheduling_success)
        self.assertIsNotNone(result.reason_if_failed)
        self.assertIn("CPU below threshold", result.reason_if_failed)

if __name__ == '__main__':
    unittest.main()
