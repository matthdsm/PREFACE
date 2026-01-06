from pathlib import Path
from typing import List, Tuple, Dict
import logging
import numpy.typing as npt
import onnx
import pandas as pd
import statsmodels.api as sm
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA
import onnx
from onnx import TensorProto, helper
from onnx.compose import add_prefix, merge_models
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from tensorflow.keras import Model  # type: ignore
from xgboost import XGBRegressor

from preface.lib.neural import neural_export
from preface.lib.svm import svm_export
from preface.lib.xgboost import xgboost_export


def preprocess_ratios(ratios_df: pd.DataFrame, exclude_chrs: list[str]) -> pd.DataFrame:
    """Preprocess ratios DataFrame by excluding chromosomes, adding region column and transposing.
    returns a x by 1 dataframe with regions as columns.
    """
    # sanitize columns
    ratios_df = ratios_df[["chr", "start", "end", "ratio"]].copy()
    # santize chr column
    ratios_df["chr"] = ratios_df["chr"].astype(str).str.replace("chr", "", regex=False)
    # exclude chromosomes
    ratios_df = ratios_df[~ratios_df["chr"].isin(exclude_chrs)].copy()
    # add region column
    ratios_df["region"] = (
        ratios_df["chr"]
        + ":"  # type: ignore
        + ratios_df["start"].astype(str)
        + "-"
        + ratios_df["end"].astype(str)
    )  # type: ignore
    # drop chr, start, end columns
    ratios_df.drop(columns=["chr", "start", "end"], inplace=True)
    # set region as index and transpose
    ratios_df = ratios_df.set_index("region").T

    return ratios_df


def fit_rlm(x_values: npt.NDArray, y_values: npt.NDArray) -> tuple[float, float]:
    """
    Fit a Robust Linear Model (RLM) using Huber's T norm.

    Args:
        x_values: The independent variables.
        y_values: The dependent variable.

    Returns:
        A tuple containing the intercept and slope of the fitted model.
    """
    # Add a constant to the independent variable array for intercept calculation
    x_rlm = sm.add_constant(x_values)

    # Fit the RLM model
    # M=sm.robust.norms.HuberT() specifies the robust norm to use for fitting,
    # which is less sensitive to outliers than ordinary least squares.
    rlm_model = sm.RLM(y_values, x_rlm, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()

    intercept, slope = rlm_results.params
    return float(intercept), float(slope)


def _ensure_opset(
    model_proto: onnx.ModelProto, version: int = 19, ml_version: int = 3
) -> onnx.ModelProto:
    """
    Force the default domain opset to a specific version.
    Also force "ai.onnx.ml" to specific version (default 3) to prevent mismatches.
    """
    for op in model_proto.opset_import:
        if (not op.domain or op.domain == "ai.onnx") and op.version != version:
            op.version = version
        if op.domain == "ai.onnx.ml" and op.version != ml_version:
            op.version = ml_version

    # Check if ai.onnx.ml is missing but we are forcing it? No, only if present.
    # But checking if we need to add it is complex, simpler to just normalize existing ones.
    return model_proto


def pca_export(pca: PCA, input_dim: int) -> onnx.ModelProto:
    """
    Convert a PCA model to ONNX format.
    """
    pca_initial_type = [("pca_input", FloatTensorType([None, input_dim]))]
    pca_onnx = convert_sklearn(
        pca,
        initial_types=pca_initial_type,
        target_opset=18,
    )
    _ensure_opset(pca_onnx, 18)  # type: ignore

    return pca_onnx  # type: ignore


def _export_processing_pipeline(
    imputer: SimpleImputer | KNNImputer | IterativeImputer,
    pca: PCA,
    input_dim: int,
) -> Tuple[onnx.ModelProto, str]:
    """
    Exports the Imputer -> PCA pipeline to ONNX.
    Returns the merged model and its output name.
    """
    # 1. Convert Imputer
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    imputer_onnx = convert_sklearn(imputer, initial_types=initial_type, target_opset=18)
    _ensure_opset(imputer_onnx, 18, ml_version=3)  # type: ignore

    # Prefix Imputer
    imputer_onnx = add_prefix(imputer_onnx, prefix="imputer_")  # type: ignore
    imputer_out_name = imputer_onnx.graph.output[0].name  # type: ignore

    # 2. Convert PCA
    pca_onnx = pca_export(pca, input_dim)
    pca_in_name = pca_onnx.graph.input[0].name  # type: ignore

    # Merge Imputer + PCA
    pipeline_model = merge_models(
        imputer_onnx,  # type: ignore
        pca_onnx,  # type: ignore
        io_map=[(imputer_out_name, pca_in_name)],  # type: ignore
    )

    return pipeline_model, pipeline_model.graph.output[0].name


def _merge_and_save_ensemble(
    prefixed_models: List[onnx.ModelProto],
    reg_names: List[str],
    input_dim: int,
    output_path: Path,
    metadata: Dict[str, str] | None = None,
) -> None:
    """
    Merges all split models, adds an averaging node, and saves the final ensemble.
    """
    logging.info(f"Merging {len(prefixed_models)} splits into final ensemble...")
    # Start with the first split
    combined_model = prefixed_models[0]

    for i in range(1, len(prefixed_models)):
        logging.debug(f"Merging split {i}...")
        combined_model = merge_models(
            combined_model,
            prefixed_models[i],
            io_map=[],
        )

    graph = combined_model.graph

    # Find all inputs that look like "split_X_imputer_input"
    split_inputs = []
    for node in graph.input:
        if "imputer_input" in node.name:  # Robust check
            split_inputs.append(node.name)

    # Create a global input
    global_input_name = "input"

    # Replace all usages of split inputs with global input
    for node in graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name in split_inputs:
                node.input[idx] = global_input_name

    # Reset graph inputs to just the global input
    while len(graph.input) > 0:
        graph.input.pop()

    graph.input.extend(
        [
            helper.make_tensor_value_info(
                global_input_name,
                FloatTensorType([None, input_dim]).to_onnx_type().tensor_type.elem_type,
                [None, input_dim],
            )
        ]
    )

    # Add Mean node
    final_reg_name = "final_ff_score"
    final_std_name = "final_ff_stdev"

    if reg_names:
        # 1. Mean(E[x])
        mean_reg_node = helper.make_node(
            "Mean", inputs=reg_names, outputs=[final_reg_name], name="Mean_FF"
        )
        graph.node.append(mean_reg_node)

        # 2. Stdev = Sqrt(E[x^2] - (E[x])^2)
        # 2a. Squared inputs x^2
        sq_names = []
        for name in reg_names:
            sq_name = name + "_sq"
            sq_names.append(sq_name)
            node = helper.make_node("Mul", inputs=[name, name], outputs=[sq_name])
            graph.node.append(node)

        # 2b. Mean of squares E[x^2]
        mean_sq_name = "mean_sq"
        node = helper.make_node(
            "Mean", inputs=sq_names, outputs=[mean_sq_name], name="Mean_Sq_FF"
        )
        graph.node.append(node)

        # 2c. Square of mean (E[x])^2
        sq_mean_name = "sq_mean"
        node = helper.make_node(
            "Mul",
            inputs=[final_reg_name, final_reg_name],
            outputs=[sq_mean_name],
            name="Sq_Mean_FF",
        )
        graph.node.append(node)

        # 2d. Variance = E[x^2] - (E[x])^2
        var_name = "variance"
        node = helper.make_node(
            "Sub", inputs=[mean_sq_name, sq_mean_name], outputs=[var_name], name="Var_FF"
        )
        graph.node.append(node)

        # 2e. Relu to ensure non-negative variance (float errors)
        var_relu_name = "variance_relu"
        node = helper.make_node(
            "Relu", inputs=[var_name], outputs=[var_relu_name], name="Var_Relu_FF"
        )
        graph.node.append(node)

        # 2f. Sqrt -> Stdev
        node = helper.make_node(
            "Sqrt", inputs=[var_relu_name], outputs=[final_std_name], name="Stdev_FF"
        )
        graph.node.append(node)

    # Clean outputs
    while len(graph.output) > 0:
        graph.output.pop()

    new_outputs = []
    if reg_names:
        new_outputs.append(
            helper.make_tensor_value_info(final_reg_name, TensorProto.FLOAT, [None, 1])
        )
        new_outputs.append(
            helper.make_tensor_value_info(final_std_name, TensorProto.FLOAT, [None, 1])
        )

    graph.output.extend(new_outputs)

    # Add metadata
    if metadata:
        for key, value in metadata.items():
            meta = combined_model.metadata_props.add()
            meta.key = key
            meta.value = value

    onnx.save(combined_model, output_path)
    logging.info(f"Ensemble saved to {output_path}")


def _export_neural_ensemble(models, input_dim, output_path, metadata):
    prefixed_models = []
    reg_names = []

    for i, (imputer, pca, model) in enumerate(models):
        logging.debug(f"Exporting neural split {i}...")
        split_prefix = f"split_{i}_"

        # Pipeline: Imputer -> PCA
        pipeline_model, pipeline_out = _export_processing_pipeline(
            imputer, pca, input_dim
        )

        # Model: Neural
        model_onnx = neural_export(model)
        _ensure_opset(model_onnx, 18, ml_version=3)

        model_prefix = "model_"
        model_onnx = add_prefix(model_onnx, prefix=model_prefix)
        model_in = model_onnx.graph.input[0].name

        # Merge
        combined_split = merge_models(
            pipeline_model, model_onnx, io_map=[(pipeline_out, model_in)]
        )

        # Track output
        # Neural output assumed to be first
        reg_base = model_prefix + model_onnx.graph.output[0].name.replace(
            model_prefix, ""
        )

        reg_name = split_prefix + model_onnx.graph.output[0].name
        reg_names.append(reg_name)

        prefixed_models.append(add_prefix(combined_split, prefix=split_prefix))

    _merge_and_save_ensemble(
        prefixed_models, reg_names, input_dim, output_path, metadata
    )


def _export_xgboost_ensemble(models, input_dim, output_path, metadata):
    prefixed_models = []
    reg_names = []

    for i, (imputer, pca, model) in enumerate(models):
        logging.debug(f"Exporting xgboost split {i}...")
        split_prefix = f"split_{i}_"

        pipeline_model, pipeline_out = _export_processing_pipeline(
            imputer, pca, input_dim
        )

        model_onnx = xgboost_export(model)
        _ensure_opset(model_onnx, 18, ml_version=3)

        model_prefix = "model_"
        model_onnx = add_prefix(model_onnx, prefix=model_prefix)
        model_in = model_onnx.graph.input[0].name

        combined_split = merge_models(
            pipeline_model, model_onnx, io_map=[(pipeline_out, model_in)]
        )

        reg_name = split_prefix + model_onnx.graph.output[0].name
        reg_names.append(reg_name)

        prefixed_models.append(add_prefix(combined_split, prefix=split_prefix))

    _merge_and_save_ensemble(
        prefixed_models, reg_names, input_dim, output_path, metadata
    )


def _export_svm_ensemble(models, input_dim, output_path, metadata):
    prefixed_models = []
    reg_names = []

    for i, (imputer, pca, model) in enumerate(models):
        logging.debug(f"Exporting svm split {i}...")
        split_prefix = f"split_{i}_"

        pipeline_model, pipeline_out = _export_processing_pipeline(
            imputer, pca, input_dim
        )

        model_onnx = svm_export(model)
        _ensure_opset(model_onnx, 18, ml_version=3)

        model_prefix = "model_"
        model_onnx = add_prefix(model_onnx, prefix=model_prefix)
        model_in = model_onnx.graph.input[0].name

        combined_split = merge_models(
            pipeline_model, model_onnx, io_map=[(pipeline_out, model_in)]
        )

        reg_name = split_prefix + model_onnx.graph.output[0].name
        reg_names.append(reg_name)

        prefixed_models.append(add_prefix(combined_split, prefix=split_prefix))

    _merge_and_save_ensemble(
        prefixed_models, reg_names, input_dim, output_path, metadata
    )


def ensemble_export(
    models: List[
        Tuple[
            SimpleImputer | KNNImputer | IterativeImputer,
            PCA,
            Model | XGBRegressor | SVR,
        ]
    ],
    input_dim: int,
    output_path: Path,
    metadata: Dict[str, str] | None = None,
) -> None:
    """
    Save an ensemble of models (Imputer + PCA + Model) combined.
    models: List of (Imputer, PCA, Model) tuples.
    metadata: Optional dictionary of metadata to save in the ONNX model.
    """
    if not models:
        return

    # Detect model type from first split
    _, _, first_model = models[0]

    if isinstance(first_model, Model):
        logging.info("Exporting Neural Network Ensemble...")
        _export_neural_ensemble(models, input_dim, output_path, metadata)
    elif isinstance(first_model, XGBRegressor):
        logging.info("Exporting XGBoost Ensemble...")
        _export_xgboost_ensemble(models, input_dim, output_path, metadata)
    elif isinstance(first_model, SVR):
        logging.info("Exporting SVM Ensemble...")
        _export_svm_ensemble(models, input_dim, output_path, metadata)
    else:
        raise ValueError(f"Unsupported model type: {type(first_model)}")
