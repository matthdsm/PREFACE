#!/usr/bin/env python3
"""
Script to summarize training metrics from multiple runs.
Recursively looks for 'training_split_metrics.csv' in the given directory (default: output).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

def main():
    parser = argparse.ArgumentParser(description="Summarize metrics.")
    parser.add_argument(
        "--root", 
        type=Path, 
        default=Path("output"), 
        help="Root directory to search (default: output)"
    )
    args = parser.parse_args()

    root_dir = args.root
    if not root_dir.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    # Find all metrics files
    metric_files = list(root_dir.rglob("training_split_metrics.csv"))
    
    if not metric_files:
        print("No 'training_split_metrics.csv' files found.")
        return

    results = []
    
    # Columns to aggregate
    agg_cols = ["mae", "rmse", "r2", "intercept", "slope"]

    for f in metric_files:
        # Directory name as label (e.g. neural_knn)
        # assuming structure output/MODEL_IMPUTE/training_split_metrics.csv
        model_name = f.parent.name
        
        try:
            df = pd.read_csv(f)
            
            # Check if columns exist
            cols = [c for c in agg_cols if c in df.columns]
            
            if not cols:
                continue
                
            stats = {}
            for c in cols:
                stats[f"{c}_mean"] = df[c].mean()
                stats[f"{c}_median"] = df[c].median()
                stats[f"{c}_std"] = df[c].std()
            
            stats["model"] = model_name
            results.append(stats)
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not results:
        print("No valid data found.")
        return

    # Create summary DF
    summary_df = pd.DataFrame(results)
    
    # Set model as first col
    cols = ["model"] + [c for c in summary_df.columns if c != "model"]
    summary_df = summary_df[cols]
    
    # Sort by model name
    summary_df = summary_df.sort_values("model")

    # Display with Rich (force width to avoid collapsing)
    console = Console(width=200)
    table = Table(title="Training Metrics Summary", show_lines=True)

    table.add_column("Model", style="cyan", no_wrap=True)
    
    for metric in agg_cols:
        table.add_column(f"{metric}\n(mean±std)", justify="center")
        table.add_column(f"{metric}\n(med)", justify="center")

    for _, row in summary_df.iterrows():
        # Prepare row data
        row_data = [str(row["model"])]
        
        for metric in agg_cols:
            if f"{metric}_mean" in row:
                mean_val = row[f"{metric}_mean"]
                std_val = row[f"{metric}_std"]
                med_val = row[f"{metric}_median"]
                
                row_data.append(f"{mean_val:.4f} ± {std_val:.4f}")
                row_data.append(f"{med_val:.4f}")
            else:
                row_data.append("-")
                row_data.append("-")
        
        table.add_row(*row_data)

    console.print(table)
    
    # Also print simpler version for piping
    print("\nPandas View:")
    print(summary_df.to_string(index=False))
    
    # Also save to CSV
    out_csv = root_dir / "summary_metrics.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSummary saved to {out_csv}")

    # --- Boxplot Generation ---
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Collect raw data for plotting
        raw_data_list = []
        for f in metric_files:
            model_name = f.parent.name
            try:
                df = pd.read_csv(f)
                # Keep only relevant columns
                cols = [c for c in agg_cols if c in df.columns]
                if not cols:
                    continue
                df_subset = df[cols].copy()
                df_subset["model"] = model_name
                raw_data_list.append(df_subset)
            except:
                pass

        if raw_data_list:
            all_data = pd.concat(raw_data_list, ignore_index=True)
            all_data = all_data.sort_values("model")
            
            # Separate plot per metric
            for metric in agg_cols:
                if metric not in all_data.columns:
                    continue
                
                plt.figure(figsize=(12, 6))
                ax = sns.boxplot(data=all_data, x="model", y=metric, hue="model", palette="Set3", legend=True)
                
                plt.title(f"{metric} Distribution by Model")
                plt.ylabel(metric)
                plt.xlabel("Model")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.xticks(rotation=45)
                
                try:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                except Exception:
                    pass

                out_plot = root_dir / f"summary_boxplot_{metric}.png"
                plt.savefig(out_plot, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Boxplot saved to {out_plot}")
            
    except ImportError:
        print("\nSkipping boxplot generation (matplotlib/seaborn not installed).")
    except Exception as e:
        print(f"\nError generating boxplot: {e}")

if __name__ == "__main__":
    main()
