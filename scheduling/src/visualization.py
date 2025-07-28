"""
Visualization module for creating plots of simulation results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from pathlib import Path

def plot_comparison_metric(results_df: pd.DataFrame, 
                          metric_name: str, 
                          output_path: Optional[Path] = None,
                          group_by: str = 'algorithm',
                          title: Optional[str] = None,
                          fig_size: tuple = (12, 6)):
    """
    Generate a bar chart comparing different algorithms or policies on a specific metric.
    
    Args:
        results_df: DataFrame with simulation results and metrics
        metric_name: Name of the metric to plot (must be a column in results_df)
        output_path: Path to save the figure (if None, just displays)
        group_by: Column to group by ('algorithm' or 'policy')
        title: Custom title for the plot (if None, generates a default)
        fig_size: Size of the figure (width, height) in inches
        
    Returns:
        None (displays or saves plot)
    """
    # Check if the metric exists in the DataFrame
    if metric_name not in results_df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in results DataFrame")
    
    # Group the data
    if group_by not in results_df.columns:
        raise ValueError(f"Group by column '{group_by}' not found in results DataFrame")
    
    # Create figure and axis
    plt.figure(figsize=fig_size)
    
    # Group by the specified column and calculate mean of the metric
    grouped_data = results_df.groupby([group_by, 'policy'])[metric_name].mean().unstack()
    
    # Create a bar chart
    ax = grouped_data.plot(kind='bar')
    
    # Add labels and title
    plt.xlabel(group_by.capitalize())
    plt.ylabel(metric_name.replace('_', ' ').title())
    
    if title is None:
        title = f"Comparison of {metric_name.replace('_', ' ').title()} by {group_by.capitalize()} and Policy"
    plt.title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add a grid to help with readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(title='Policy')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path is not None:
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
        
    # Close the plot to free memory
    plt.close()

def plot_individual_schedule_details(machine_df: pd.DataFrame,
                                    start_time: pd.Timestamp,
                                    end_time: pd.Timestamp,
                                    title: str = None,
                                    output_path: Optional[Path] = None,
                                    fig_size: tuple = (14, 8)):
    """
    Plot the details of a specific scheduled workload period.
    
    Args:
        machine_df: DataFrame with the machine data
        start_time: Start time of the scheduled period
        end_time: End time of the scheduled period
        title: Title for the plot
        output_path: Path to save the figure (if None, just displays)
        fig_size: Size of the figure (width, height) in inches
        
    Returns:
        None (displays or saves plot)
    """
    # Extract the window data
    window_data = machine_df.loc[start_time:end_time].copy()
    
    # Create figure with two subplots (CPU and Solar)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, sharex=True)
    
    # Plot CPU data
    ax1.plot(window_data.index, window_data['cpu_actual'], label='Actual CPU', color='blue')
    ax1.plot(window_data.index, window_data['cpu_predicted'], label='Predicted CPU', color='orange', linestyle='--')
    ax1.set_ylabel('CPU Utilization (%)')
    ax1.set_title('CPU Utilization during Scheduled Window' if title is None else title)
    ax1.grid(True)
    ax1.legend()
    
    # Plot Solar data
    ax2.plot(window_data.index, window_data['solar_pv_aligned'], label='Solar PV', color='green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Solar Power')
    ax2.set_title('Solar PV Availability during Scheduled Window')
    ax2.grid(True)
    ax2.legend()
    
    # Format the x-axis to show dates clearly
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save or show the plot
    if output_path is not None:
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
        
    # Close the plot to free memory
    plt.close()

def plot_heatmap_comparison(results_df: pd.DataFrame,
                           metric_name: str,
                           x_axis: str = 'algorithm',
                           y_axis: str = 'policy',
                           output_path: Optional[Path] = None,
                           title: Optional[str] = None,
                           fig_size: tuple = (10, 8)):
    """
    Generate a heatmap comparing different algorithms and policies on a specific metric.
    
    Args:
        results_df: DataFrame with simulation results and metrics
        metric_name: Name of the metric to plot (must be a column in results_df)
        x_axis: Column to use for x-axis
        y_axis: Column to use for y-axis
        output_path: Path to save the figure (if None, just displays)
        title: Custom title for the plot (if None, generates a default)
        fig_size: Size of the figure (width, height) in inches
        
    Returns:
        None (displays or saves plot)
    """
    # Check if required columns exist
    for col in [metric_name, x_axis, y_axis]:
        if col not in results_df.columns:
            raise ValueError(f"Column '{col}' not found in results DataFrame")
    
    # Create a pivot table for the heatmap
    pivot_data = results_df.pivot_table(
        values=metric_name,
        index=y_axis,
        columns=x_axis,
        aggfunc='mean'
    )
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
    
    # Add title
    if title is None:
        title = f"Heatmap of {metric_name.replace('_', ' ').title()} by {x_axis.capitalize()} and {y_axis.capitalize()}"
    plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path is not None:
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
        
    # Close the plot to free memory
    plt.close()
