from pathlib import Path
from typing import List, Tuple, Dict

import onnx
from onnx import TensorProto, helper
from onnx.compose import add_prefix, merge_models
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from tensorflow.keras import Model  # type: ignore
from xgboost import XGBRegressor

from preface.lib.neural import neural_export
from preface.lib.svm import svm_export
from preface.lib.xgboost import xgboost_export


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
        current_model = merge_models(
            imputer_onnx,  # type: ignore
            pca_onnx,  # type: ignore
            io_map=[(imputer_out_name, pca_in_name)],  # type: ignore
        )

        # 3. Convert Model
        model_type = "unknown"
        if isinstance(model, Model):
            model_type = "nn"
            current_model = neural_export(model)

        elif isinstance(model, SVR):
            current_model = svm_export(model)

        elif isinstance(model, XGBRegressor):
            model_type = "xgb"
            current_model = xgboost_export(model)

        # Prefix everything in this split's graph
        prefixed_model = add_prefix(current_model, prefix=split_prefix)
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
    # Because we added "imputer_" prefix inside the loop, and then "split_i_" prefix outside.
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
    class_names = []

    # Helper to find output names
    for i, (model_type, reg_base, class_base) in enumerate(split_info):
        prefix = f"split_{i}_"

        if model_type == "nn":
            reg_names.append(prefix + reg_base)
            if class_base:
                class_names.append(prefix + class_base)

        elif model_type == "xgb":
            # XGBoost output [Batch, 2] -> Col 0: Reg, Col 1: Class (Prob)
            xgb_out = prefix + reg_base
            split_reg = f"{prefix}reg_split"
            split_class = f"{prefix}class_split"

            split_node = helper.make_node(
                "Split",
                inputs=[xgb_out],
                outputs=[split_reg, split_class],
                name=f"{prefix}Split",
                axis=1,
                split=[1, 1],
            )
            graph.node.append(split_node)

            reg_names.append(split_reg)
            class_names.append(split_class)

        elif model_type == "svm":
            reg_names.append(prefix + reg_base)

            # SVC prob output [Batch, 2] -> want col 1
            svm_class_prob_out = prefix + class_base
            svm_class_split = f"{prefix}class_prob_split"
            svm_class_0_dummy = f"{prefix}class_prob_0_dummy"

            split_node_svm = helper.make_node(
                "Split",
                inputs=[svm_class_prob_out],
                outputs=[svm_class_0_dummy, svm_class_split],
                name=f"{prefix}SVMSplit",
                axis=1,
                split=[1, 1],
            )
            graph.node.append(split_node_svm)

            class_names.append(svm_class_split)

    # Add Mean nodes
    final_reg_name = "final_ff_score"
    final_class_name = "final_sex_prob"

    # Check if we have outputs to average
    if reg_names:
        mean_reg_node = helper.make_node(
            "Mean", inputs=reg_names, outputs=[final_reg_name], name="Mean_FF"
        )
        graph.node.append(mean_reg_node)

    if class_names:
        mean_class_node = helper.make_node(
            "Mean", inputs=class_names, outputs=[final_class_name], name="Mean_Sex"
        )
        graph.node.append(mean_class_node)

    # Clean outputs
    while len(graph.output) > 0:
        graph.output.pop()

    new_outputs = []
    if reg_names:
        new_outputs.append(
            helper.make_tensor_value_info(final_reg_name, TensorProto.FLOAT, [None, 1])
        )
    if class_names:
        new_outputs.append(
            helper.make_tensor_value_info(
                final_class_name, TensorProto.FLOAT, [None, 1]
            )
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
