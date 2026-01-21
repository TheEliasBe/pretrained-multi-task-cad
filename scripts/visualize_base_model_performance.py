import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich import box
from rich.console import Console
from rich.table import Table

import os
import wandb

# Set plotting style
plt.style.use("default")
sns.set_palette("husl")

# Initialize wandb API
api = wandb.Api()

# Project details - set these environment variables or update here
entity = os.environ.get("WANDB_ENTITY", "your-wandb-entity")
project = os.environ.get("WANDB_PROJECT", "cadcoder-seqcompl")

print(f"Connecting to wandb project: {entity}/{project}")
print(f"Full URL: https://wandb.ai/{entity}/{project}")

# Fetch all runs from the project
runs = api.runs(f"{entity}/{project}")

print(f"Found {len(runs)} runs in the project")
print("Fetching run data...")

# Convert runs to a list for easier processing
runs_list = list(runs)
print(f"Successfully loaded {len(runs_list)} runs")

# Extract data from all runs
run_data = []

for run in runs_list:
    # Get basic run information
    run_info = {
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "runtime": run.summary.get("_runtime", None),
        "tags": run.tags,
        "group": run.group,
        "job_type": run.job_type,
    }

    # Add all summary metrics
    for key, value in run.summary.items():
        if not key.startswith("_"):  # Skip private wandb fields
            run_info[key] = value

    # Add config parameters
    for key, value in run.config.items():
        run_info[f"config_{key}"] = value

    run_data.append(run_info)

# Create DataFrame
df = pd.DataFrame(run_data)
print(f"Created DataFrame with {len(df)} runs and {len(df.columns)} columns")
print(f"DataFrame shape: {df.shape}")

# Display basic info about the runs
print(f"\nRun states: {df['state'].value_counts().to_dict()}")
print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")

# Display DataFrame overview
print("DataFrame Info:")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nFirst few rows:")
print("=" * 50)
# Display first few rows (limit columns for readability)
display_cols = ["run_id", "name", "state", "created_at"]
if len(display_cols) <= len(df.columns):
    print(df[display_cols].head())
else:
    print(df.head())

print("\nColumn names:")
print("=" * 50)
for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

# Analysis: Find minimum validation/generation loss by model name
print("\n" + "=" * 80)
print("ANALYSIS: Minimum Validation/Generation Loss by Model")
print("=" * 80)

# First, let's identify which columns contain validation or generation loss
loss_columns = [
    col
    for col in df.columns
    if "loss" in col.lower() and ("val" in col.lower() or "generation" in col.lower())
]
print(f"Found potential validation/generation loss columns: {loss_columns}")

# Add additional validation metrics for analysis
additional_val_metrics = ["val/chamfer", "val/codebleu", "val/invalid_ratio", "val/nc"]
available_additional_metrics = [
    col for col in additional_val_metrics if col in df.columns
]
print(f"Found additional validation metrics: {available_additional_metrics}")

# Combine all metrics for analysis
all_metrics_to_analyze = loss_columns + available_additional_metrics
print(f"All metrics to analyze: {all_metrics_to_analyze}")

# Also check for any loss columns that might be relevant
all_loss_columns = [col for col in df.columns if "loss" in col.lower()]
print(f"All loss-related columns: {all_loss_columns}")

# Debug: Check if the additional metrics exist in the columns
print("Checking for additional metrics in DataFrame columns:")
for metric in additional_val_metrics:
    exists = metric in df.columns
    print(f"  {metric}: {'✅ Found' if exists else '❌ Not found'}")

# Show all validation columns for reference
val_columns = [col for col in df.columns if col.startswith("val/")]
print(f"All validation columns available: {val_columns}")

# Check if we have model name in config
model_config_cols = [
    col for col in df.columns if "config" in col.lower() and "model" in col.lower()
]
print(f"Model-related config columns: {model_config_cols}")

# Let's also check the cadcoder config which might contain model info
print("\nChecking config_cadcoder column for model information...")
if "config_cadcoder" in df.columns:
    # Sample some non-null cadcoder configs to see the structure
    sample_configs = df["config_cadcoder"].dropna().head(3)
    print("Sample config_cadcoder values:")
    for i, config in enumerate(sample_configs):
        print(f"{i+1}. {config}")

    # If configs are dictionaries, try to extract model_name
    if len(sample_configs) > 0:
        first_config = sample_configs.iloc[0]
        if isinstance(first_config, dict) and "model_name" in first_config:
            print("\n✅ Found model_name in config_cadcoder!")
            # Extract model names for all runs
            df["model_name"] = df["config_cadcoder"].apply(
                lambda x: x.get("model_name", None) if isinstance(x, dict) else None
            )
            model_col = "model_name"
        else:
            print(f"\n❌ config_cadcoder structure: {type(first_config)}")
            if isinstance(first_config, dict):
                print(f"Available keys: {list(first_config.keys())}")
else:
    print("config_cadcoder column not found")

# Also check if there are other potential model identifiers
print("\nLooking for other potential model identifiers...")
for col in df.columns:
    if (
        "model" in col.lower()
        or "architecture" in col.lower()
        or "backbone" in col.lower()
    ):
        print(f"Potential model column: {col}")

# Focus on finished runs only for meaningful comparison
finished_runs = df[df["state"] == "finished"].copy()
print(f"\nAnalyzing {len(finished_runs)} finished runs out of {len(df)} total runs")

# Check if we have the model_name column we extracted
model_col = None
if "model_name" in df.columns:
    model_col = "model_name"
    print("Using extracted model_name column")
elif len(model_config_cols) > 0:
    model_col = model_config_cols[0]
    print(f"Using model configuration column: {model_col}")

if len(finished_runs) > 0 and model_col is not None:
    # Show unique model names
    unique_models = finished_runs[model_col].dropna().unique()
    print(f"Unique models found: {list(unique_models)}")

    # Store results for the summary table
    summary_results = {}

    # For each metric, find best performing model
    for metric_col in all_metrics_to_analyze:
        if metric_col in finished_runs.columns:
            print(f"\n--- Analysis for {metric_col} ---")

            # Filter runs that have both model name and metric value
            valid_runs = finished_runs.dropna(subset=[model_col, metric_col])

            if len(valid_runs) > 0:
                # Determine if lower or higher is better for this metric
                is_lower_better = any(
                    keyword in metric_col.lower()
                    for keyword in ["loss", "invalid_ratio", "chamfer"]
                )

                # Group by model and find aggregated stats
                metric_stats = (
                    valid_runs.groupby(model_col)[metric_col]
                    .agg(["min", "max", "mean", "count"])
                    .reset_index()
                )
                metric_stats.columns = [
                    model_col,
                    f"min_{metric_col}",
                    f"max_{metric_col}",
                    f"mean_{metric_col}",
                    "run_count",
                ]

                if is_lower_better:
                    # Sort by minimum (ascending - lower is better)
                    metric_stats = metric_stats.sort_values(f"min_{metric_col}")
                    sort_desc = "lower is better"
                    best_value_col = f"min_{metric_col}"
                else:
                    # Sort by maximum (descending - higher is better)
                    metric_stats = metric_stats.sort_values(
                        f"max_{metric_col}", ascending=False
                    )
                    sort_desc = "higher is better"
                    best_value_col = f"max_{metric_col}"

                print(
                    f"Results (sorted by {'minimum' if is_lower_better else 'maximum'} value, {sort_desc}):"
                )
                print("-" * 60)
                for _, row in metric_stats.iterrows():
                    print(f"Model: {row[model_col]}")
                    print(f"  Min {metric_col}: {row[f'min_{metric_col}']:.6f}")
                    print(f"  Max {metric_col}: {row[f'max_{metric_col}']:.6f}")
                    print(f"  Avg {metric_col}: {row[f'mean_{metric_col}']:.6f}")
                    print(f"  # Runs: {row['run_count']}")
                    print()

                # Find the best model for this metric
                best_model = metric_stats.iloc[0]
                print(f"🏆 BEST MODEL for {metric_col}: {best_model[model_col]}")
                print(f"   Best Value: {best_model[best_value_col]:.6f} ({sort_desc})")
                print(f"   Based on {best_model['run_count']} runs")

                # Store result for summary table
                summary_results[metric_col] = {
                    "best_model": best_model[model_col],
                    "best_value": best_model[best_value_col],
                    "run_count": best_model["run_count"],
                    "is_lower_better": is_lower_better,
                }

                # Find the specific run that achieved this best value
                if is_lower_better:
                    best_run = valid_runs[
                        (valid_runs[model_col] == best_model[model_col])
                        & (valid_runs[metric_col] == best_model[f"min_{metric_col}"])
                    ].iloc[0]
                else:
                    best_run = valid_runs[
                        (valid_runs[model_col] == best_model[model_col])
                        & (valid_runs[metric_col] == best_model[f"max_{metric_col}"])
                    ].iloc[0]

                print(f"   Best run ID: {best_run['run_id']}")
                print(f"   Best run name: {best_run['name']}")
                print()
            else:
                print(f"No valid data found for {metric_col}")

    # Create a beautiful summary table using rich
    console = Console()

    print("\n" + "=" * 80)
    print("📊 SUMMARY: Best Models by Metric")
    print("=" * 80)

    if summary_results:
        # Create the summary table
        table = Table(title="🏆 Best Model Performance Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Best Model", style="green")
        table.add_column("Best Value", style="yellow", justify="right")
        table.add_column("# Runs", style="blue", justify="center")
        table.add_column("Direction", style="magenta", justify="center")

        # Add rows for each metric
        for metric, result in summary_results.items():
            direction = "↓ Lower" if result["is_lower_better"] else "↑ Higher"
            # Shorten model names for better display
            model_name = (
                result["best_model"].split("/")[-1]
                if "/" in result["best_model"]
                else result["best_model"]
            )
            table.add_row(
                metric,
                model_name,
                f"{result['best_value']:.6f}",
                str(result["run_count"]),
                direction,
            )

        console.print(table)

        # Save summary table as CSV
        summary_data = []
        for metric, result in summary_results.items():
            direction = "Lower" if result["is_lower_better"] else "Higher"
            model_name = (
                result["best_model"].split("/")[-1]
                if "/" in result["best_model"]
                else result["best_model"]
            )
            summary_data.append(
                {
                    "Metric": metric,
                    "Best_Model": model_name,
                    "Best_Value": result["best_value"],
                    "Num_Runs": result["run_count"],
                    "Direction": direction,
                    "Full_Model_Name": result["best_model"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = "model_performance_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n💾 Summary table saved to: {summary_csv_path}")

        # Create a model comparison table
        print("\n")
        comparison_table = Table(title="🔍 Model Comparison Matrix", box=box.ROUNDED)
        comparison_table.add_column("Model", style="cyan")

        # Add columns for each metric
        for metric in summary_results.keys():
            short_metric = metric.replace("val/", "").replace("train/", "")
            comparison_table.add_column(short_metric, justify="right")

        # Get all unique models
        all_models = list(unique_models)

        # For each model, show its performance across all metrics
        for model in all_models:
            row_data = [model.split("/")[-1] if "/" in model else model]

            for metric in summary_results.keys():
                # Find this model's performance for this metric
                valid_runs = finished_runs.dropna(subset=[model_col, metric])
                model_runs = valid_runs[valid_runs[model_col] == model]

                if len(model_runs) > 0:
                    is_lower_better = summary_results[metric]["is_lower_better"]
                    if is_lower_better:
                        best_value = model_runs[metric].min()
                    else:
                        best_value = model_runs[metric].max()

                    # Highlight if this is the best model for this metric
                    if model == summary_results[metric]["best_model"]:
                        row_data.append(f"[bold green]{best_value:.4f}[/bold green]")
                    else:
                        row_data.append(f"{best_value:.4f}")
                else:
                    row_data.append("-")

            comparison_table.add_row(*row_data)

        console.print(comparison_table)

        # Save comparison table as CSV
        comparison_data = []
        for model in all_models:
            row_data = {
                "Model": model.split("/")[-1] if "/" in model else model,
                "Full_Model_Name": model,
            }

            for metric in summary_results.keys():
                # Find this model's performance for this metric
                valid_runs = finished_runs.dropna(subset=[model_col, metric])
                model_runs = valid_runs[valid_runs[model_col] == model]

                if len(model_runs) > 0:
                    is_lower_better = summary_results[metric]["is_lower_better"]
                    if is_lower_better:
                        best_value = model_runs[metric].min()
                    else:
                        best_value = model_runs[metric].max()

                    # Clean metric name for column
                    clean_metric = metric.replace("val/", "").replace("train/", "")
                    row_data[clean_metric] = best_value

                    # Mark if this is the best model for this metric
                    row_data[f"{clean_metric}_is_best"] = (
                        model == summary_results[metric]["best_model"]
                    )
                else:
                    clean_metric = metric.replace("val/", "").replace("train/", "")
                    row_data[clean_metric] = None
                    row_data[f"{clean_metric}_is_best"] = False

            comparison_data.append(row_data)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv_path = "model_comparison_matrix.csv"
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"\n💾 Comparison matrix saved to: {comparison_csv_path}")

        # Create a detailed results CSV with all metrics for all models
        detailed_data = []
        for model in all_models:
            for metric in summary_results.keys():
                valid_runs = finished_runs.dropna(subset=[model_col, metric])
                model_runs = valid_runs[valid_runs[model_col] == model]

                if len(model_runs) > 0:
                    detailed_data.append(
                        {
                            "Model": model.split("/")[-1] if "/" in model else model,
                            "Full_Model_Name": model,
                            "Metric": metric,
                            "Min_Value": model_runs[metric].min(),
                            "Max_Value": model_runs[metric].max(),
                            "Mean_Value": model_runs[metric].mean(),
                            "Std_Value": model_runs[metric].std(),
                            "Num_Runs": len(model_runs),
                            "Is_Best_Model": (
                                model == summary_results[metric]["best_model"]
                            ),
                            "Metric_Direction": "Lower"
                            if summary_results[metric]["is_lower_better"]
                            else "Higher",
                        }
                    )

        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv_path = "detailed_model_metrics.csv"
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"💾 Detailed metrics saved to: {detailed_csv_path}")

        print("\n📁 All CSV files saved in current directory:")
        print(f"   1. {summary_csv_path} - Best model per metric")
        print(f"   2. {comparison_csv_path} - Model comparison matrix")
        print(f"   3. {detailed_csv_path} - All metrics for all models")
    else:
        print("No metrics analyzed.")

else:
    print("❌ Cannot perform analysis:")
    if len(finished_runs) == 0:
        print("- No finished runs found")
    if model_col is None:
        print("- No model name column found")

print("\n" + "=" * 80)
