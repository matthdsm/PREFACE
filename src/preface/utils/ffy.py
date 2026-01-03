"""
FFY calculation utility.
"""

from pathlib import Path
import numpy as np
import typer


def wisecondorx_ffy(
    wisecondorx_npz: Path = typer.Argument(
        ...,
        help="Path to WisecondorX NPZ file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    sex_cutoff: float = typer.Option(
        0.2, "--sex-cutoff", help="Cutoff for sex determination"
    ),
    slope: float = typer.Option(
        1.0, "--slope", help="Slope for FFY calculation (Y = slope * FF + intercept)"
    ),
    intercept: float = typer.Option(
        0.0, "--intercept", help="Intercept for FFY calculation"
    ),
) -> dict:
    """
    Calculate fetal fraction from Y chromosome (FFY) using WisecondorX output.
    FFY is calculated as: (Y_fraction - intercept) / slope
    """
    npz = np.load(wisecondorx_npz, encoding="latin1", allow_pickle=True)
    read_counts = npz["sample"].item()
    # Calculate read depth from autosomes (1-22) + X (23) + Y (24)?
    # The original code did range(1, 25) which is 1..24.
    # Usually 23 is X, 24 is Y.
    read_depth = float(
        np.sum([np.sum(read_counts[x]) for x in [str(y) for y in range(1, 25)]])
    )
    y_chr_fraction = np.array(read_counts["24"], dtype="float") / read_depth

    y_frac_total = np.sum(y_chr_fraction)

    # Calculate FFY using linear regression formula
    if slope == 0:
        ffy_val = 0.0 # Avoid division by zero
    else:
        ffy_val = (y_frac_total - intercept) / slope

    sex = "unknown"
    if y_frac_total > sex_cutoff:  # Sex determination usually based on raw Y fraction
        sex = "male"
    else:
        sex = "female"

    return {"FFY": ffy_val, "sex": sex, "Y_fraction": y_frac_total}
