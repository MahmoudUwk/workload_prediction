import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import inspect  # NEW

# Add the project directory to the path so modules can be imported
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
try:
    from src.visualization import plot_comparison_metric, plot_heatmap_comparison
    from config.scheduling_config import LOG_LEVEL
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure that the script is run from the project root or that the PYTHONPATH is set correctly.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def find_latest_run_directory(base_output_dir: Path) -> Path | None:
    """Finds the most recent run_YYYYMMDD_HHMMSS directory."""
    run_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        logger.warning(f"No run directories found in {base_output_dir}")
        return None
    latest_run_dir = max(run_dirs, key=lambda d: d.name)
    logger.info(f"Found latest run directory: {latest_run_dir}")
    return latest_run_dir

def load_and_combine_results(latest_run_dir: Path) -> pd.DataFrame | None:
    """Loads and combines averaged results from all relevant dataset files."""
    data_dir = latest_run_dir / "data"
    logger.info(f"Scanning for machine averaged result files in: {data_dir}")

    result_files = list(data_dir.glob("*_machine_averaged_results.csv"))
    
    if not result_files:
        logger.warning(f"No machine averaged result files found in {data_dir} matching pattern '*_machine_averaged_results.csv'.")
        return None

    all_dfs = []
    for file_path in result_files:
        try:
            # Extract dataset name from filename, e.g., "Alibaba_SMALL" from "Alibaba_SMALL_machine_averaged_results.csv"
            dataset_name_full = file_path.name.replace("_machine_averaged_results.csv", "")
            
            logger.info(f"Loading results from: {file_path} for dataset: {dataset_name_full}")
            df = pd.read_csv(file_path)
            df['dataset'] = dataset_name_full  # Use the full name like "Alibaba_SMALL"
            all_dfs.append(df)
            logger.info(f"Successfully loaded and tagged {len(df)} rows from {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading or processing file {file_path}: {e}")

    if not all_dfs:
        logger.error("No data loaded from any files. Cannot combine results.")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined DataFrame created with {len(combined_df)} rows from {len(all_dfs)} files.")
    return combined_df

def create_output_directory_for_combined_analysis(base_output_dir: Path) -> Path:
    """Creates a timestamped directory for combined analysis outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir_name = f"combined_analysis_{timestamp}"
    analysis_output_dir = base_output_dir / analysis_dir_name
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    (analysis_output_dir / "plots").mkdir(exist_ok=True)
    logger.info(f"Created output directory for combined analysis: {analysis_output_dir}")
    return analysis_output_dir

def generate_visualizations(combined_df: pd.DataFrame, output_plot_dir: Path):
    """Generates and saves bar-charts with all load sizes side-by-side."""
    if combined_df.empty:
        logger.warning("Combined DataFrame is empty. Skipping visualizations.")
        return

    # ------------------------------------------------------------------
    # 1. Ensure 'Dataset Source' and 'Load Size' columns exist
    if 'Dataset Source' not in combined_df.columns or 'Load Size' not in combined_df.columns:
        parsed = combined_df['dataset'].str.rsplit('_', n=1, expand=True)
        combined_df['Dataset Source'] = parsed[0]
        combined_df['Load Size'] = parsed[1]
    # ------------------------------------------------------------------

    # 2. Metrics to keep (max two plots per dataset-source)
    metrics_to_plot = ['energy_consumed_wh', 'geuf_percentage']

    for metric in metrics_to_plot:
        if metric not in combined_df.columns:
            logger.warning(f"Metric '{metric}' not in DataFrame. Skipping plot.")
            continue

        for ds_source, ds_group in combined_df.groupby('Dataset Source'):
            try:
                plot_title = (f"{metric.replace('_', ' ').title()} • {ds_source} "
                              f"(all load sizes)")
                # base filename without extension
                plot_base = output_plot_dir / f"{ds_source}_{metric}_by_loadsize"

                for ext in ("png", "eps"):
                    plot_path = plot_base.with_suffix(f".{ext}")

                    # Group bars by load size (one figure per data-source)
                    plot_comparison_metric(
                        results_df=ds_group,
                        metric_name=metric,
                        output_path=plot_path,
                        title=plot_title,
                        group_by='Load Size'
                    )
                logger.info(f"Saved {metric} plots (png & eps) with all load sizes for {ds_source}")
            except Exception as e:
                logger.error(f"Error plotting {metric} for {ds_source}: {e}")

    logger.info("Heat-map generation skipped to limit plot count.")
    logger.info(f"Visualizations saved to: {output_plot_dir}")

# ---------- Styler compatibility helpers  ----------
def _hide_index_compat(styler):
    """Hide index across pandas versions, returning a valid Styler."""
    if hasattr(styler, "hide"):
        try:                       # pandas ≥ 1.4
            result = styler.hide(axis="index", inplace=False)
            return result if result is not None else styler
        except TypeError:          # pandas < 1.4 signature
            styler.hide(axis="index")
            return styler
    if hasattr(styler, "hide_index"):  # very old pandas
        styler.hide_index()
    return styler


def _styler_to_latex_compat(styler, **kwargs):
    """
    Call Styler.to_latex using only the kwargs supported by the current pandas
    version to avoid TypeError for unknown parameters.
    """
    valid_params = inspect.signature(styler.to_latex).parameters
    safe_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return styler.to_latex(**safe_kwargs)
# ---------------------------------------------------

def generate_latex_tables(combined_df: pd.DataFrame, analysis_output_dir: Path):
    """Generates separate LaTeX tables for each dataset source, with abbreviated policies."""
    logger.info("Generating LaTeX tables for each dataset source.")

    if combined_df.empty:
        logger.warning("Combined DataFrame is empty. Skipping LaTeX table generation.")
        return
    
    if 'dataset' not in combined_df.columns:
        logger.error("'dataset' column not found in combined_df. Cannot parse Dataset Source and Load Size.")
        return

    table_df = combined_df.copy()

    try:
        parsed_dataset_info = table_df['dataset'].str.rsplit('_', n=1, expand=True)
        table_df['Dataset Source'] = parsed_dataset_info[0]
        table_df['Load Size'] = parsed_dataset_info[1]
        load_order = ['SMALL', 'MEDIUM', 'LARGE']
        table_df['Load Size'] = pd.Categorical(table_df['Load Size'], categories=load_order, ordered=True)
        logger.info("Successfully parsed 'Dataset Source' and 'Load Size'.")
    except Exception as e:
        logger.error(f"Error parsing 'Dataset Source' and 'Load Size' from 'dataset' column: {e}")
        return

    # Columns retained for LaTeX tables (dropped: avg_cpu_actual, grid_energy_wh)
    metric_columns_ordered = [
        'energy_consumed_wh',
        'geuf_percentage',
        'green_energy_wh'
    ]
    metric_columns_for_table = [m for m in metric_columns_ordered if m in table_df.columns]

    if not metric_columns_for_table:
        logger.warning("No relevant metric columns found for LaTeX table generation. Skipping.")
        return

    # Minimise energy consumed; maximise green-related metrics
    min_metrics_config = ['energy_consumed_wh']
    max_metrics_config = ['geuf_percentage', 'green_energy_wh']

    renamed_cols_map = {col: col.replace('_', ' ').title() for col in metric_columns_for_table}
    if 'geuf_percentage' in renamed_cols_map:
        renamed_cols_map['geuf_percentage'] = 'GEUF (\%)'

    policy_abbreviations = {
        'PREDICTIVE_CARBON_AWARE': 'PCA',
        'NAIVE_CARBON_AWARE': 'NCA',
        'GREEN_OBLIVIOUS': 'GO'
        # Add other policy names if they exist and need abbreviation
    }

    for dataset_source_name, source_group_df in table_df.groupby('Dataset Source'):
        logger.info(f"Generating LaTeX table for dataset source: {dataset_source_name}")
        
        current_table_df = source_group_df.copy()
        
        # Drop 'Dataset Source' (implicit) and 'algorithm' columns
        current_table_df.drop(columns=['Dataset Source', 'algorithm'], inplace=True, errors='ignore')
        
        # Abbreviate and rename 'policy' column
        current_table_df['policy'] = current_table_df['policy'].replace(policy_abbreviations)
        current_table_df.rename(columns={'policy': 'Policy'}, inplace=True)

        policy_col_name = 'Policy' if 'Policy' in current_table_df.columns else 'policy'
        
        display_columns_ordered = ['Load Size', policy_col_name] + metric_columns_for_table
        actual_display_columns = [col for col in display_columns_ordered if col in current_table_df.columns or col in renamed_cols_map]
        
        missing_display_cols = [col for col in actual_display_columns if col not in current_table_df.columns and col not in renamed_cols_map and col not in ['Load Size', policy_col_name]]

        if missing_display_cols:
            logger.error(f"Missing essential metric columns for table display for {dataset_source_name}: {missing_display_cols}. Aborting LaTeX table for this dataset.")
            continue
            
        table_df_display = current_table_df[[col for col in actual_display_columns if col in current_table_df.columns or col in renamed_cols_map]].copy()
        table_df_display.rename(columns=renamed_cols_map, inplace=True)
        
        table_df_display.sort_values(by=['Load Size', policy_col_name], inplace=True)

        renamed_min_metrics = [renamed_cols_map.get(m) for m in min_metrics_config if renamed_cols_map.get(m) in table_df_display.columns]
        renamed_max_metrics = [renamed_cols_map.get(m) for m in max_metrics_config if renamed_cols_map.get(m) in table_df_display.columns]

        # Create a new DataFrame that will contain formatted strings, ready for LaTeX commands
        table_df_latex = pd.DataFrame(index=table_df_display.index, columns=table_df_display.columns)

        # Populate table_df_latex with formatted strings
        for col in table_df_display.columns:
            if col in renamed_cols_map.values():
                numeric_series = pd.to_numeric(table_df_display[col], errors='coerce')
                table_df_latex[col] = numeric_series.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            else:
                table_df_latex[col] = table_df_display[col].astype(str)
        
        # Define the LaTeX command for highlighting
        for load_s in table_df_display['Load Size'].unique():
            load_specific_rows_indices = table_df_display[table_df_display['Load Size'] == load_s].index
            
            for metric_col in renamed_min_metrics:
                if metric_col in table_df_display.columns:
                    metric_series_numeric = pd.to_numeric(table_df_display.loc[load_specific_rows_indices, metric_col], errors='coerce')
                    if not metric_series_numeric.empty and not metric_series_numeric.isnull().all():
                        min_val = metric_series_numeric.min()
                        for idx in load_specific_rows_indices:
                            current_val_numeric = pd.to_numeric(table_df_display.loc[idx, metric_col], errors='coerce')
                            if pd.notna(current_val_numeric) and abs(current_val_numeric - min_val) < 1e-9:
                                formatted_val_str = table_df_latex.loc[idx, metric_col]
                                table_df_latex.loc[idx, metric_col] = rf"\highlightcell{{{formatted_val_str}}}"
            
            for metric_col in renamed_max_metrics:
                if metric_col in table_df_display.columns:
                    metric_series_numeric = pd.to_numeric(table_df_display.loc[load_specific_rows_indices, metric_col], errors='coerce')
                    if not metric_series_numeric.empty and not metric_series_numeric.isnull().all():
                        max_val = metric_series_numeric.max()
                        for idx in load_specific_rows_indices:
                            current_val_numeric = pd.to_numeric(table_df_display.loc[idx, metric_col], errors='coerce')
                            if pd.notna(current_val_numeric) and abs(current_val_numeric - max_val) < 1e-9:
                                formatted_val_str = table_df_latex.loc[idx, metric_col]
                                table_df_latex.loc[idx, metric_col] = rf"\highlightcell{{{formatted_val_str}}}"

        styler = table_df_latex.style
        styler = _hide_index_compat(styler)  # UPDATED

        num_metrics = len([m for m in metric_columns_for_table if renamed_cols_map.get(m,m) in table_df_latex.columns])
        col_widths = ["1.8cm", "1.5cm"] + ["2.0cm"] * num_metrics 
        final_column_format = "|" + "|".join([f"m{{{w}}}" for w in col_widths]) + "|"
        
        table_label = f"tab:{dataset_source_name.lower().replace(' ', '_')}_results_summary"
        # --- improved caption ------------------------------------------------
        caption_parts = [
            f"Machine-averaged sustainability metrics for {dataset_source_name} "
            f"(grouped by load size and scheduling policy)."
        ]

        desc_segments = []
        if renamed_min_metrics:
            desc_segments.append(
                f"↓ lower is better for {', '.join(renamed_min_metrics)}"
            )
        if renamed_max_metrics:
            desc_segments.append(
                f"↑ higher is better for {', '.join(renamed_max_metrics)}"
            )

        if desc_segments:
            caption_parts.append(
                "Highlighted cells indicate the best value per load size "
                f"({' ; '.join(desc_segments)})."
            )

        caption_text = " ".join(caption_parts)
        # ---------------------------------------------------------------------

        try:
            latex_txt_intermediate = _styler_to_latex_compat(   # UPDATED
                styler,
                column_format=final_column_format,
                position="h!tbp",
                position_float="centering",
                hrules=True,
                label=table_label,
                caption=caption_text,
                multirow_align="c",
                multicol_align="c",
                escape=False
            )

            latex_lines = latex_txt_intermediate.splitlines(True)
            final_latex_lines = []
            
            header_found = False
            midrule_after_header_idx = -1
            for idx, line in enumerate(latex_lines):
                if "\\" in line and not header_found:
                    header_found = True
                if header_found and "\\midrule" in line and midrule_after_header_idx == -1:
                    midrule_after_header_idx = idx
                    break
            
            if midrule_after_header_idx != -1:
                final_latex_lines.extend(latex_lines[:midrule_after_header_idx + 1])
                
                for df_row_idx in range(len(table_df_display)):
                    if df_row_idx > 0 and \
                       table_df_display.iloc[df_row_idx]['Load Size'] != table_df_display.iloc[df_row_idx - 1]['Load Size']:
                        final_latex_lines.append('\\midrule\n')
                    
                    current_data_row_line_idx = midrule_after_header_idx + 1 + df_row_idx
                    if current_data_row_line_idx < len(latex_lines):
                        final_latex_lines.append(latex_lines[current_data_row_line_idx])
                    else:
                        logger.warning(f"LaTeX line index out of bounds for data row {df_row_idx} in {dataset_source_name}. Table might be incomplete.")
                        break 
                
                remaining_lines_start_idx = midrule_after_header_idx + 1 + len(table_df_display)
                if remaining_lines_start_idx < len(latex_lines):
                    final_latex_lines.extend(latex_lines[remaining_lines_start_idx:])
                else:
                    logger.warning(f"Potentially missing tail of LaTeX table for {dataset_source_name}.")

                latex_txt = "".join(final_latex_lines)
            else:
                logger.warning(f"Could not find midrule after header for {dataset_source_name}. Using intermediate LaTeX output.")
                latex_txt = latex_txt_intermediate

            # ---------------------------------------------------------
            # Ensure the \highlightcell macro is defined to avoid the
            # "Undefined control sequence" LaTeX error.
            # It is inserted only if the macro or a previous definition
            # is not already present in the generated snippet.
            if "\\highlightcell" in latex_txt and "\\newcommand{\\highlightcell}" not in latex_txt:
                macro_def = "\\newcommand{\\highlightcell}[1]{\\textbf{#1}}\n\n"
                latex_txt = macro_def + latex_txt
            # ---------------------------------------------------------

            output_latex_file = analysis_output_dir / f"table_{dataset_source_name}_summary_results.tex"
            with open(output_latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_txt)
            logger.info(f"LaTeX table for {dataset_source_name} saved to: {output_latex_file}")

        except Exception as e:
            logger.error(f"Error generating LaTeX table for {dataset_source_name}: {e}")
            logger.error(f"Problematic table_df_display for {dataset_source_name}:\\n{table_df_display.head().to_string()}")

def main():
    """Main function to load, combine, and visualize results."""
    logger.info("Starting combined analysis script.")
    
    base_output_dir = project_root / "output"
    
    latest_run_dir = find_latest_run_directory(base_output_dir)
    if not latest_run_dir:
        logger.error("Could not find the latest run directory. Exiting.")
        return

    combined_df = load_and_combine_results(latest_run_dir)
    if combined_df is None or combined_df.empty:
        logger.error("Failed to load or combine data. Exiting.")
        return

    analysis_output_dir = create_output_directory_for_combined_analysis(base_output_dir)
    combined_csv_path = analysis_output_dir / "combined_machine_averaged_results.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    logger.info(f"Combined DataFrame saved to: {combined_csv_path}")

    plots_dir = analysis_output_dir / "plots"
    generate_visualizations(combined_df, plots_dir)
    
    generate_latex_tables(combined_df, analysis_output_dir)
    
    logger.info("Combined analysis script finished.")

if __name__ == "__main__":
    main()
