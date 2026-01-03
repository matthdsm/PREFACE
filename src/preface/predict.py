"""
Predict module for PREFACE.
"""

import pandas as pd
import typer
from preface.lib.functions import preprocess_ratios
from rich import print
from pathlib import Path
import onnxruntime as ort
import numpy as np


def preface_predict(
    infile: Path = typer.Option(..., "--infile", help="Path to input BED file"),
    model_path: Path = typer.Option(..., "--model", help="Path to model"),
) -> None:
    """
    Predict using model.
    """

    # Load model
    preface_model: ort.InferenceSession = ort.InferenceSession(model_path)

    # Check metadata for excluded chromosomes
    meta = preface_model.get_modelmeta()
    custom_props = meta.custom_metadata_map

    if "exclude_chrs" in custom_props:
        exclude_str = custom_props["exclude_chrs"]
        if exclude_str:
            exclude_chrs = exclude_str.split(",")
        else:
            exclude_chrs = []
        print(
            f"[dim]Using excluded chromosomes from model metadata: {exclude_chrs}[/dim]"
        )

    ratios: pd.DataFrame = pd.read_csv(infile, sep="\t")

    # Preprocess ratios
    preprocessed_ratios = preprocess_ratios(ratios, exclude_chrs=exclude_chrs)

    # Convert to float32
    input_data = preprocessed_ratios.values.astype(np.float32)

    # Run inference
    # We can rely on position 0, 1 or names.
    # build_ensemble saves: final_ff_score, final_sex_prob

    # Get output names
    output_names = [o.name for o in preface_model.get_outputs()]

    # Run
    results = preface_model.run(
        output_names, {preface_model.get_inputs()[0].name: input_data}
    )

    # Map results to meaningful variables
    # If standard PREFACE model, names are specific.
    # If unknown model, fallback to index.

    ff_score = None

    result_map = dict(zip(output_names, results))

    if "final_ff_score" in result_map:
        ff_score = result_map["final_ff_score"][0][0]
    elif len(results) > 0:
        ff_score = results[0][0][0]

    print("--- Patient Report ---")
    print(f"Predicted FF Score: {ff_score:.4f}")
