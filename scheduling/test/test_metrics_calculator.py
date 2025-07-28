"""
Unit tests for the metrics calculator module.
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
from src.metrics_calculator import (
    calculate_simulation_metrics, calculate_prediction_accuracy, aggregate_metrics
)
from src.simulation_engine import SimulationResult, SchedulingPolicy

class TestMetricsCalculator(unittest.TestCase):
    """Test cases for the metrics calculator."""
    
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
        
        # Define a test simulation result
        # Choose a window during daylight hours for realistic solar values
        mid_day_idx = len(self.test_df) // 2
        self.test_window_start = self.test_df.index[mid_day_idx]
        self.test_window_end = self.test_df.index[mid_day_idx + 12]  # 12 x 5min = 1 hour
        
        window_data = self.test_df.loc[self.test_window_start:self.test_window_end]
        
        self.test_simulation_result = SimulationResult(
            start_time=self.test_window_start,
            end_time=self.test_window_end,
            machine_id="test_machine",
            algorithm="test_algo",
            policy=SchedulingPolicy.PREDICTIVE_CARBON_AWARE,
            avg_cpu_actual=window_data['cpu_actual'].mean(),
            avg_cpu_predicted=window_data['cpu_predicted'].mean(),
            avg_solar_pv=window_data['solar_pv_aligned'].mean(),
            workload_type="TEST",
            max_cpu_during_workload=window_data['cpu_actual'].max(),
            min_cpu_during_workload=window_data['cpu_actual'].min(),
            scheduling_success=True,
            reason_if_failed=None
        )
        
        # Define test workload config
        self.test_workload = {
            'cores_required': 2,
            'duration_minutes': 60,
            'min_cpu_utilization': 20,
        }
        
        # Define test power model
        self.test_power_model = {
            'idle_power_watts': 100,
            'max_power_watts': 300,
            'power_per_core_watts': 25,
            'utilization_to_power_curve': 'linear',
        }
    
    def test_calculate_simulation_metrics(self):
        """Test calculation of simulation metrics."""
        metrics = calculate_simulation_metrics(
            self.test_simulation_result,
            self.test_df,
            self.test_workload,
            self.test_power_model
        )
        
        # Check that all expected metrics are present
        self.assertIn('energy_consumed_wh', metrics)
        self.assertIn('green_energy_wh', metrics)
        self.assertIn('grid_energy_wh', metrics)
        self.assertIn('geuf_percentage', metrics)
        self.assertIn('scheduling_success', metrics)
        
        # Check that metrics are reasonable values
        self.assertTrue(metrics['scheduling_success'])
        self.assertIsNone(metrics['failure_reason'])
        self.assertGreater(metrics['energy_consumed_wh'], 0)
        self.assertGreaterEqual(metrics['green_energy_wh'], 0)
        self.assertGreaterEqual(metrics['grid_energy_wh'], 0)
        self.assertLessEqual(metrics['geuf_percentage'], 100)
        self.assertGreaterEqual(metrics['geuf_percentage'], 0)
        
        # Sum of green and grid energy should equal total energy
        self.assertAlmostEqual(
            metrics['green_energy_wh'] + metrics['grid_energy_wh'],
            metrics['energy_consumed_wh'],
            delta=1e-6  # Allow small floating-point error
        )
    
    def test_calculate_metrics_failed_simulation(self):
        """Test calculating metrics for a failed simulation."""
        failed_simulation = SimulationResult(
            start_time=None,
            end_time=None,
            machine_id="test_machine",
            algorithm="test_algo",
            policy=SchedulingPolicy.PREDICTIVE_CARBON_AWARE,
            avg_cpu_actual=None,
            avg_cpu_predicted=None,
            avg_solar_pv=None,
            workload_type="TEST",
            max_cpu_during_workload=None,
            min_cpu_during_workload=None,
            scheduling_success=False,
            reason_if_failed="No viable windows found"
        )
        
        metrics = calculate_simulation_metrics(
            failed_simulation,
            self.test_df,
            self.test_workload,
            self.test_power_model
        )
        
        # Check that the metrics reflect the failure
        self.assertFalse(metrics['scheduling_success'])
        self.assertEqual(metrics['failure_reason'], "No viable windows found")
        self.assertIsNone(metrics['energy_consumed_wh'])
        self.assertIsNone(metrics['green_energy_wh'])
        self.assertIsNone(metrics['grid_energy_wh'])
        self.assertIsNone(metrics['geuf_percentage'])
    
    def test_calculate_prediction_accuracy(self):
        """Test calculation of prediction accuracy metrics."""
        # Use the first 100 points for testing
        actual = self.test_df['cpu_actual'].iloc[:100]
        predicted = self.test_df['cpu_predicted'].iloc[:100]
        
        accuracy = calculate_prediction_accuracy(actual, predicted)
        
        # Check that all expected metrics are present
        self.assertIn('MAE', accuracy)
        self.assertIn('RMSE', accuracy)
        self.assertIn('MAPE', accuracy)
        
        # Check that metrics are reasonable values
        self.assertGreaterEqual(accuracy['MAE'], 0)
        self.assertGreaterEqual(accuracy['RMSE'], 0)
        self.assertGreaterEqual(accuracy['MAPE'], 0)
        
        # MAE should be less than or equal to RMSE
        self.assertLessEqual(accuracy['MAE'], accuracy['RMSE'])
        
        # Calculate metrics manually to verify
        mae_expected = np.mean(np.abs(actual - predicted))
        rmse_expected = np.sqrt(np.mean((actual - predicted) ** 2))
        
        self.assertAlmostEqual(accuracy['MAE'], mae_expected, places=6)
        self.assertAlmostEqual(accuracy['RMSE'], rmse_expected, places=6)
    
    def test_aggregate_metrics(self):
        """Test aggregation of metrics from multiple simulations."""
        # Create multiple simulation results
        simulation_results = []
        metrics_results = []
        
        # Add the existing test simulation
        simulation_results.append(self.test_simulation_result)
        metrics_results.append(calculate_simulation_metrics(
            self.test_simulation_result,
            self.test_df,
            self.test_workload,
            self.test_power_model
        ))
        
        # Create a second simulation with different policy
        second_sim = SimulationResult(
            start_time=self.test_window_start + timedelta(hours=2),
            end_time=self.test_window_end + timedelta(hours=2),
            machine_id="test_machine",
            algorithm="test_algo",
            policy=SchedulingPolicy.NAIVE_CARBON_AWARE,
            avg_cpu_actual=45.0,
            avg_cpu_predicted=48.0,
            avg_solar_pv=80.0,
            workload_type="TEST",
            max_cpu_during_workload=55.0,
            min_cpu_during_workload=35.0,
            scheduling_success=True,
            reason_if_failed=None
        )
        
        simulation_results.append(second_sim)
        metrics_results.append(calculate_simulation_metrics(
            second_sim,
            self.test_df,
            self.test_workload,
            self.test_power_model
        ))
        
        # Add a failed simulation
        failed_sim = SimulationResult(
            start_time=None,
            end_time=None,
            machine_id="test_machine",
            algorithm="test_algo",
            policy=SchedulingPolicy.GREEN_OBLIVIOUS,
            avg_cpu_actual=None,
            avg_cpu_predicted=None,
            avg_solar_pv=None,
            workload_type="TEST",
            max_cpu_during_workload=None,
            min_cpu_during_workload=None,
            scheduling_success=False,
            reason_if_failed="No viable windows"
        )
        
        simulation_results.append(failed_sim)
        metrics_results.append({
            "energy_consumed_wh": None,
            "green_energy_wh": None,
            "grid_energy_wh": None,
            "geuf_percentage": None,
            "scheduling_success": False,
            "failure_reason": "No viable windows"
        })
        
        # Aggregate the metrics
        agg_df = aggregate_metrics(simulation_results, metrics_results)
        
        # Check that the DataFrame is not empty and has the expected columns
        self.assertFalse(agg_df.empty)
        for col in ['algorithm', 'policy', 'machine_id', 'workload_type', 'geuf_percentage']:
            self.assertIn(col, agg_df.columns)
        
        # Check that we have the expected number of rows (only successful simulations)
        self.assertEqual(len(agg_df), 2)
        
        # Check that the policies are correct
        policies = agg_df['policy'].unique()
        self.assertEqual(len(policies), 2)
        self.assertIn(SchedulingPolicy.PREDICTIVE_CARBON_AWARE.name, policies)
        self.assertIn(SchedulingPolicy.NAIVE_CARBON_AWARE.name, policies)
        
        # GREEN_OBLIVIOUS should not appear as it was a failed simulation
        self.assertNotIn(SchedulingPolicy.GREEN_OBLIVIOUS.name, policies)

if __name__ == '__main__':
    unittest.main()
