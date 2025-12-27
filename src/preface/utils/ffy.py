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
) -> dict:
    """
    Calculate fetal fraction from Y chromosome (FFY) using WisecondorX output.
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

    ffy_val = np.sum(y_chr_fraction)

    sex = "unknown"
    if ffy_val > sex_cutoff:
        sex = "male"
    else:
        sex = "female"

    return {"FFY": ffy_val, "sex": sex}
