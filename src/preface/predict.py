"""
Predict module for PREFACE.
"""

import pandas as pd
import typer
from tensorflow.keras import load_model  # pylint: disable=no-name-in-module,import-error # type: ignore
from preface.lib.schemas import SampleDataSchema
from preface.lib.functions import preprocess_ratios


def preface_predict(
    infile: str = typer.Option(..., "--infile", help="Path to input BED file"),
    model_path: str = typer.Option(..., "--model", help="Path to model"),
) -> None:
    """
    Predict using model.
    """

    # Load model
    preface_model = load_model(model_path)
    ratios = pd.read_csv(infile, sep="\t")

    # Validate input data
    try:
        SampleDataSchema().validate(ratios)
    except Exception as e:
        typer.echo(f"Validation error: {e}")
        raise

    # Preprocess ratios
    preprocessed_ratios = preprocess_ratios(ratios, exclude_chrs=[])

    # x_bins = ratios[ratios["chr"] == "X"]
    # x_ratio: float
    # if len(x_bins) > 0:
    #     x_ratio = float(2 ** np.mean(x_bins["ratio"].dropna()))
    # else:
    #     x_ratio = float(np.nan)

    ff_pred, sex_pred = preface_model.predict(preprocessed_ratios.values)
    typer.echo(f"FF = {ff_pred:.4g}%")
    typer.echo(f"Sex = {sex_pred}")

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

    # prediction: float
    # if is_olm:
    #     prediction = float(model.predict(projected_ratio)[0])
    # else:
    #     prediction = float(model.predict(projected_ratio).flatten()[0])

    # prediction = the_intercept + the_slope * prediction

    # json_dict = {"FFX": ffx / 100, "PREFACE": prediction / 100}

    # if json_output:
    #     if json_output not in ("stdout", ""):
    #         with open(json_output, "w", encoding="utf-8") as f:
    #             json.dump(json_dict, f)
    #     else:
    #         typer.echo(json.dumps(json_dict))
    # else:
    #     typer.echo(f"FFX = {ffx:.4g}%")
    #     typer.echo(f"PREFACE = {prediction:.4g}%")
