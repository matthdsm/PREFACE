import os
import sys
from typing import List

import numpy as np
import pandas as pd
import typer


def _convert_single_npz(npz_path: str, output_dir: str) -> None:
    """
    Converts a single NPZ file to one or more Parquet files.
    """
    try:
        npz_data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        typer.echo(f"Error loading {npz_path}: {e}", err=True)
        return

    base_name = os.path.splitext(os.path.basename(npz_path))[0]

    typer.echo(f"Processing NPZ file: {npz_path}")

    for key in npz_data.files:
        array = npz_data[key]

        # Determine output filename
        output_filename = f"{base_name}_{key}.parquet"
        output_filepath = os.path.join(output_dir, output_filename)

        typer.echo(
            f"  Converting array '{key}' (shape: {array.shape}, dtype: {array.dtype}) to {output_filepath}"
        )

        # Handle different array dimensions
        df: pd.DataFrame
        if array.ndim == 1:
            # 1D array, convert to a single-column DataFrame
            df = pd.DataFrame({key: array})
        elif array.ndim == 2:
            # 2D array, convert to DataFrame where columns are named 'key_0', 'key_1', etc.
            # Or if it's a structured array, pandas can handle it directly.
            if array.dtype.fields:  # Check if it's a structured numpy array
                df = pd.DataFrame(array)
            else:
                df = pd.DataFrame(
                    array, columns=[f"{key}_{i}" for i in range(array.shape[1])]
                )
        elif array.ndim > 2:
            typer.echo(
                f"  Warning: Array '{key}' has {array.ndim} dimensions. Flattening for Parquet storage.",
                err=True,
            )
            # Flatten to 1D and then treat as a single-column DataFrame
            df = pd.DataFrame({key: array.flatten()})
        else:  # Scalar case (ndim == 0)
            df = pd.DataFrame({key: [array.item()]})  # Store as a single-row DataFrame

        try:
            df.to_parquet(output_filepath, index=False)
            typer.echo(f"  Successfully saved '{key}' to {output_filepath}")
        except Exception as e:
            typer.echo(f"  Error saving array '{key}' to Parquet: {e}", err=True)


def npz_to_parquet(
    npz_files: List[str] = typer.Argument(..., help="One or more .npz files to convert."),
    output_dir: str = typer.Option(
        ".", "-o", "--output-dir", help="Directory to save the output Parquet files."
    ),
) -> None:
    """
    Convert NumPy .npz files to Parquet files for easier exploration.
    """
    os.makedirs(output_dir, exist_ok=True)

    for npz_file in npz_files:
        if not os.path.exists(npz_file):
            typer.echo(f"Error: Input file not found: {npz_file}", err=True)
            continue
        _convert_single_npz(npz_file, output_dir)


if __name__ == "__main__":
    typer.run(npz_to_parquet)