"""
Predict module for PREFACE.
"""

# pylint: disable=too-many-locals,too-many-branches,too-many-statements

import os
import json
from typing import Optional, Union
import numpy as np
import pandas as pd
import joblib
import typer
from sklearn.linear_model import LinearRegression
from tensorflow import keras  # pylint: disable=no-name-in-module,import-error


def preface_predict(
    infile: str = typer.Option(..., "--infile", help="Path to input BED file"),
    model_path_base: str = typer.Option(
        ..., "--model", help="Path to model (directory or model_meta.pkl)"
    ),
    json_output: Optional[str] = typer.Option(
        None,
        "--json",
        help="Output JSON. If filename provided, writes to file. Pass 'stdout' for console output.",
    ),
) -> None:
    """
    Predict using model.
    """
    if os.path.isdir(model_path_base):
        meta_path = os.path.join(model_path_base, "model_meta.pkl")
    else:
        meta_path = model_path_base

    if not os.path.exists(meta_path):
        root, _ = os.path.splitext(model_path_base)
        if os.path.exists(root + ".pkl"):
            meta_path = root + ".pkl"
        else:
            typer.echo(f"The file '{meta_path}' does not exist.")
            raise typer.Exit(code=1)

    model_data = joblib.load(meta_path)

    n_feat = model_data["n_feat"]
    mean_features = model_data["mean_features"]
    possible_features = model_data["possible_features"]
    pca = model_data["pca"]
    is_olm = model_data["is_olm"]
    the_intercept = model_data["the_intercept"]
    the_slope = model_data["the_slope"]
    # Variable names in the pickle are fixed, but we map them to snake_case locals
    intercept_x = model_data["the_intercept_X"]
    slope_x = model_data["the_slope_X"]

    dir_path = os.path.dirname(meta_path)
    model: Union[LinearRegression, keras.Model]
    if is_olm:
        model = joblib.load(os.path.join(dir_path, "model_weights.pkl"))
    else:
        model = keras.models.load_model(os.path.join(dir_path, "model_weights.keras"))

    bin_table = pd.read_csv(infile, sep="\t")

    x_bins = bin_table[bin_table["chr"] == "X"]
    x_ratio: float
    if len(x_bins) > 0:
        x_ratio = float(2 ** np.mean(x_bins["ratio"].dropna()))
    else:
        x_ratio = float(np.nan)

    ffx: float = (x_ratio - intercept_x) / slope_x

    bin_table["feat_id"] = (
        bin_table["chr"].astype(str)
        + ":"
        + bin_table["start"].astype(str)
        + "-"
        + bin_table["end"].astype(str)
    )

    ratio_map = bin_table.set_index("feat_id")["ratio"]
    features = ratio_map.reindex(possible_features)
    features = features.fillna(mean_features)

    features_array = features.values.reshape(1, -1)

    projected_ratio = pca.transform(features_array)[:, :n_feat]

    prediction: float
    if is_olm:
        prediction = float(model.predict(projected_ratio)[0])
    else:
        prediction = float(model.predict(projected_ratio).flatten()[0])

    prediction = the_intercept + the_slope * prediction

    json_dict = {"FFX": ffx / 100, "PREFACE": prediction / 100}

    if json_output:
        if json_output not in ("stdout", ""):
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(json_dict, f)
        else:
            typer.echo(json.dumps(json_dict))
    else:
        typer.echo(f"FFX = {ffx:.4g}%")
        typer.echo(f"PREFACE = {prediction:.4g}%")
