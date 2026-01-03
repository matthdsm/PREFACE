from pathlib import Path
from typing import List, Tuple, Dict
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


def _ensure_opset(model_proto: onnx.ModelProto, version: int = 19) -> onnx.ModelProto:
    """
    Force the default domain opset to a specific version.
    """
    for op in model_proto.opset_import:
        if (not op.domain or op.domain == "ai.onnx") and op.version < version:
            op.version = version
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
    _ensure_opset(pca_onnx, 19)  # type: ignore

    return pca_onnx  # type: ignore


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

    prefixed_models = []
    split_info = []  # Store (model_type, reg_base, class_base) for each split

    # Process each split
    for i, (imputer, pca, model) in enumerate(models):
        split_prefix = f"split_{i}_"

        # 1. Convert Imputer
        initial_type = [("input", FloatTensorType([None, input_dim]))]
        imputer_onnx = convert_sklearn(
            imputer, initial_types=initial_type, target_opset=18
        )
        _ensure_opset(imputer_onnx, 19)  # type: ignore

        # Prefix Imputer
        imputer_onnx = add_prefix(imputer_onnx, prefix="imputer_")  # type: ignore
        imputer_out_name = imputer_onnx.graph.output[0].name  # type: ignore

        # 2. Convert PCA
        pca_onnx = pca_export(pca, input_dim)
        pca_in_name = pca_onnx.graph.input[0].name  # type: ignore

        # Merge Imputer + PCA
        current_model_imp_pca = merge_models(
            imputer_onnx,  # type: ignore
            pca_onnx,  # type: ignore
            io_map=[(imputer_out_name, pca_in_name)],  # type: ignore
        )

        # 3. Convert Model
        if isinstance(model, Model):
            model_type = "nn"
            current_model = neural_export(model)

        elif isinstance(model, SVR):
            model_type = "svm"
            current_model = svm_export(model)

        elif isinstance(model, XGBRegressor):
            model_type = "xgb"
            current_model = xgboost_export(model)

        # Ensure opset matches
        _ensure_opset(current_model, 19)

        # Merge (Imputer + PCA) with Model
        # output of PCA is the input to the Model
        # We need to find the output name of the current (Imputer+PCA) model
        # which is essentially the output of the PCA part.

        # current_model_imp_pca has outputs.
        # current_model has inputs.

        pca_out_name = current_model_imp_pca.graph.output[0].name
        model_in_name = current_model.graph.input[0].name

        # To avoid name collisions between PCA internal nodes and Model internal nodes
        # (e.g. "variable" is common), we should prefix the Model before merging.
        # But we need to know the new input name after prefixing.

        model_prefix = "model_"
        current_model_prefixed = add_prefix(current_model, prefix=model_prefix)
        model_in_name_prefixed = model_prefix + model_in_name

        combined_split_model = merge_models(
            current_model_imp_pca,
            current_model_prefixed,
            io_map=[(pca_out_name, model_in_name_prefixed)],
        )

        # Collect split info for averaging later
        # We need to know the output names of the model part
        # Neural, XGB, SVM have different output structures.
        if model_type == "nn":
            # Neural has "reg_output" theoretically.
            # We assume the output we want is the first one or named specifically.
            # But let's trust the graph output 0 is what we want for regression.
            reg_base = model_prefix + current_model.graph.output[0].name

        elif model_type == "xgb":
            # XGBoost: output[0] is prediction
            reg_base = model_prefix + current_model.graph.output[0].name

        elif model_type == "svm":
            # SVM: output[0] is prediction
            reg_base = model_prefix + current_model.graph.output[0].name

        split_info.append(reg_base)

        # Prefix everything in this split's graph
        prefixed_model = add_prefix(combined_split_model, prefix=split_prefix)
        prefixed_models.append(prefixed_model)

    # --- Merge All splits ---
    # We want a single input "input" that feeds into all split_i_input

    # Start with the first split
    combined_model = prefixed_models[0]

    for i in range(1, len(prefixed_models)):
        combined_model = merge_models(
            combined_model,
            prefixed_models[i],
            io_map=[],
        )

    graph = combined_model.graph

    # Find all inputs that look like "split_X_imputer_input"
    # The input name structure is likely "split_i_imputer_input"

    # We need to find the specific input names created by add_prefix
    # We'll look for any input ending with "imputer_input"
    split_inputs = []
    for node in graph.input:
        if node.name.endswith("imputer_input"):
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

    # --- Average Outputs ---
    reg_names = []

    # Helper to find output names
    for i, reg_base in enumerate(split_info):
        prefix = f"split_{i}_"
        reg_names.append(prefix + reg_base)

    # Add Mean nodes
    final_reg_name = "final_ff_score"

    # Check if we have outputs to average
    if reg_names:
        mean_reg_node = helper.make_node(
            "Mean", inputs=reg_names, outputs=[final_reg_name], name="Mean_FF"
        )
        graph.node.append(mean_reg_node)

    # Clean outputs
    while len(graph.output) > 0:
        graph.output.pop()

    new_outputs = []
    if reg_names:
        new_outputs.append(
            helper.make_tensor_value_info(final_reg_name, TensorProto.FLOAT, [None, 1])
        )

    graph.output.extend(new_outputs)

    # Add metadata
    if metadata:
        for key, value in metadata.items():
            meta = combined_model.metadata_props.add()
            meta.key = key
            meta.value = value

    onnx.save(combined_model, output_path)
    print(f"Ensemble saved to {output_path}")
