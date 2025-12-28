"""
Predict module for PREFACE.
"""

import pandas as pd
import typer
from preface.lib.functions import preprocess_ratios
from rich import print
from pathlib import Path
import onnxruntime as ort


def preface_predict(
    infile: Path = typer.Option(..., "--infile", help="Path to input BED file"),
    model_path: Path = typer.Option(..., "--model", help="Path to model"),
) -> None:
    """
    Predict using model.
    """

    # Load model
    preface_model: ort.InferenceSession = ort.InferenceSession(model_path)
    ratios: pd.DataFrame = pd.read_csv(infile, sep="\t")

    # Preprocess ratios
    preprocessed_ratios = preprocess_ratios(ratios, exclude_chrs=[])

    # x_bins = ratios[ratios["chr"] == "X"]
    # x_ratio: float
    # if len(x_bins) > 0:
    #     x_ratio = float(2 ** np.mean(x_bins["ratio"].dropna()))
    # else:
    #     x_ratio = float(np.nan)

    results = preface_model.run(None, {preface_model.get_inputs()[0].name: preprocessed_ratios.values})
    ff_score = results[0][0][0]  # type: ignore
    sex_prob = results[1][0][0]  # type: ignore
    sex_class = "Male" if sex_prob > 0.5 else "Female"

    print("--- Patient Report ---")
    print(f"Predicted FF Score: {ff_score:.4f}")
    print(f"Sex Probability:    {sex_prob:.4f} ({sex_class})")

    # ffx: float = (x_ratio - intercept_x) / slope_x

    # bin_table["feat_id"] = (
    #     bin_table["chr"].astype(str)
    #     + ":"
    #     + bin_table["start"].astype(str)
    #     + "-"
    #     + bin_table["end"].astype(str)
    # )

    # ratio_map = bin_table.set_index("feat_id")["ratio"]
    # features = ratio_map.reindex(possible_features)
    # features = features.fillna(mean_features)

    # features_array = features.values.reshape(1, -1)

    # projected_ratio = pca.transform(features_array)[:, :n_feat]

    # prediction = float(model.predict(projected_ratio).flatten()[0])

    # prediction = the_intercept + the_slope * prediction

    # json_dict = {"FFX": ffx / 100, "PREFACE": prediction / 100}

    # if json_output:
    #     if json_output not in ("stdout", ""):
    #         with open(json_output, "w", encoding="utf-8") as f:
    #             json.dump(json_dict, f)
    #     else:
    #         print(json.dumps(json_dict))
    # else:
    #     print(f"FFX = {ffx:.4g}%")
    #     print(f"PREFACE = {prediction:.4g}%")
