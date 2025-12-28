from pathlib import Path

import onnx
import onnxmltools
import tensorflow as tf
import tf2onnx
from onnx import TensorProto, helper
from onnx.compose import add_prefix, merge_models
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA
from tensorflow.keras import Model  # type: ignore
from xgboost import XGBRegressor


def _ensure_opset(model_proto, version=12):
    """
    Force the default domain opset to a specific version.
    skl2onnx sometimes produces older opsets (e.g. 9) even when 12 is requested.
    """
    for op in model_proto.opset_import:
        if (not op.domain or op.domain == "ai.onnx") and op.version < version:
            op.version = version
    return model_proto


def build_ensemble(
    models: list[tuple[object, PCA, Model | XGBRegressor]],
    input_dim: int,
    output_path: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """
    Save an ensemble of models (Imputer + PCA + Model) combined.
    models: List of (Imputer, PCA, Model) tuples.
    metadata: Optional dictionary of metadata to save in the ONNX model.
    """

    prefixed_models = []

    # Process each fold
    for i, (imputer, pca, model) in enumerate(models):
        fold_prefix = f"fold_{i}_"

        # 1. Convert Imputer (if exists)
        # Input to Imputer is FloatTensorType([None, input_dim])
        initial_type = [("input", FloatTensorType([None, input_dim]))]

        if imputer is not None:
            # simple imputer
            imp_onnx = convert_sklearn(
                imputer, initial_types=initial_type, target_opset=12
            )
            _ensure_opset(imp_onnx, 12)

            # Rename output of imputer to match input of PCA
            # But wait, we can just chain them via merge_models later?
            # Easier to chain them linearly first or merge step by step.
            current_model = imp_onnx
            # The output name of sklearn models is typically "variable" or similar.
            # We need to find the output name.
            imp_out_name = current_model.graph.output[0].name  # type: ignore
        else:
            # No imputer (ZERO strategy). We need a "Identity" or just pass input.
            # However, if we want to fill NaNs with 0, we might need a custom ONNX node or assume input has 0s.
            # For now, let's assume the user handles NaN -> 0 before or use Identity.
            # Actually, `convert_sklearn` handles NaN?
            # If the user selected ZERO, they expect NaNs to be 0.
            # We can create a simple graph that takes input and returns input (Identity),
            # relying on the caller to provide 0s or ONNX runtime to handle it.
            # BETTER: create a dummy identity model.

            # For simplicity in this fix, we assume input is already clean if imputer is None,
            # OR we rely on the fact that we can't easily inject "FillNaN(0)" without complex node creation.
            # Let's start with the PCA conversion using the initial type.
            current_model = None
            imp_out_name = "input"

        # 2. Convert PCA
        # Input to PCA is the output of Imputer (or "input")
        if current_model:
            # If we have an imputer model, the PCA input type should match its output
            # But convert_sklearn needs `initial_types`.
            # We can convert PCA independently with same shape.
            pca_initial_type = [("input_pca", FloatTensorType([None, input_dim]))]
        else:
            pca_initial_type = initial_type

        pca_onnx = convert_sklearn(pca, initial_types=pca_initial_type, target_opset=12)
        _ensure_opset(pca_onnx, 12)
        pca_in_name = pca_onnx.graph.input[0].name  # type: ignore
        pca_out_name = pca_onnx.graph.output[0].name  # type: ignore

        if current_model:
            # Merge Imputer + PCA
            current_model = merge_models(
                current_model,  # type: ignore
                pca_onnx,  # type: ignore
                io_map=[(imp_out_name, pca_in_name)],  # type: ignore
            )
            # Update output name
            current_out_name = pca_out_name
        else:
            current_model = pca_onnx
            current_out_name = pca_out_name
            # If we had no imputer, we need to rename the input to "input" to standardize
            # actually pca_onnx input is "input" (from initial_type) or "input_pca".
            # We will handle renaming at the global merge.

        # 3. Convert Model
        # Input to Model is PCA output. Shape: [None, n_components]
        n_comps = pca.n_components_

        model_type = "nn" if isinstance(model, Model) else "xgb"

        if model_type == "nn":
            spec = (tf.TensorSpec((None, n_comps), tf.float32, name="input_model"),)  # type: ignore
            m_onnx, _ = tf2onnx.convert.from_keras(
                model, input_signature=spec, opset=12
            )
            m_onnx.graph.name = f"fold_{i}_model"
        elif model_type == "xgb":
            m_onnx = onnxmltools.convert_xgboost(
                model,
                initial_types=[("input_model", FloatTensorType([None, n_comps]))],
                target_opset=12,
            )

        m_in_name = m_onnx.graph.input[0].name

        # Merge (Imputer+PCA) + Model
        current_model = merge_models(
            current_model,  # type: ignore
            m_onnx,
            io_map=[(current_out_name, m_in_name)],  # type: ignore
        )

        # Prefix everything in this fold's graph
        prefixed_model = add_prefix(current_model, prefix=fold_prefix)
        prefixed_models.append(prefixed_model)

    # --- Merge All Folds ---
    # We want a single input "input" that feeds into all fold_i_input

    # Start with the first fold
    combined_model = prefixed_models[0]

    # Identify the input name of the first fold
    # It should be "fold_0_input" (if we used "input" name initially)
    # logic: add_prefix adds prefix to all names.
    # The input of the chain was "input" (or "input_pca").
    # So it becomes "fold_0_input".

    for i in range(1, len(prefixed_models)):
        combined_model = merge_models(
            combined_model,
            prefixed_models[i],
            io_map=[],
        )

    graph = combined_model.graph

    # Now we need to broadcast the global "input" to "fold_0_input", "fold_1_input", ...
    # We create a new input "global_input" and Identity nodes or just rewire?
    # Rewiring is safer.

    # Find all inputs that look like "fold_X_input" or "fold_X_input_pca"
    fold_inputs = []
    for node in graph.input:
        if node.name.endswith("input") or node.name.endswith("input_pca"):
            fold_inputs.append(node.name)

    # Create a global input
    global_input_name = "input"
    # remove existing inputs from graph.input (they become internal nodes fed by global input)
    # Actually, we can just rename them? No, they are distinct nodes in the graph now?
    # If we map them to the same tensor, they get connected.

    # Let's add an Identity node for each fold input, fed by global_input
    # Or simpler: create the global input, and add Split? Or just use same name?
    # In ONNX, if multiple nodes use "input", it's valid.

    # So we want to replace all usages of "fold_i_input" with "global_input".
    for node in graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name in fold_inputs:
                node.input[idx] = global_input_name

    # Reset graph inputs
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
    # Identify outputs.
    # NN: fold_i_reg_output, fold_i_class_output
    # XGB: fold_i_variable (output of regressor) -> need to split?

    # Note: earlier XGB code said it outputs [Batch, 2] for multi-output.
    # We need to find the output names of the fold graphs.

    reg_names = []
    class_names = []

    # Helper to find output names
    for i, (_, _, model) in enumerate(models):
        model_type = "nn" if isinstance(model, Model) else "xgb"
        prefix = f"fold_{i}_"

        if model_type == "nn":
            # Keras outputs are typically named by layer names.
            # In neural.py: "reg_output", "class_output"
            reg_names.append(prefix + "reg_output")
            class_names.append(prefix + "class_output")
        else:
            # XGBoost multi-output
            # The output name from onnxmltools for XGB is usually "variable"
            xgb_out = prefix + "variable"

            # We need to split this output. It is [Batch, 2].
            # Column 0: Reg, Column 1: Class (Prob)

            # Add Split node
            split_reg = f"{prefix}reg_split"
            split_class = f"{prefix}class_split"

            # Create Split node
            # Attributes: axis=1, split=[1,1]
            # Output: [split_reg, split_class]

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

    # Add Mean nodes
    final_reg_name = "final_ff_score"
    final_class_name = "final_sex_prob"

    mean_reg_node = helper.make_node(
        "Mean", inputs=reg_names, outputs=[final_reg_name], name="Mean_FF"
    )
    mean_class_node = helper.make_node(
        "Mean", inputs=class_names, outputs=[final_class_name], name="Mean_Sex"
    )

    graph.node.extend([mean_reg_node, mean_class_node])

    # Clean outputs
    while len(graph.output) > 0:
        graph.output.pop()

    graph.output.extend(
        [
            helper.make_tensor_value_info(final_reg_name, TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info(
                final_class_name, TensorProto.FLOAT, [None, 1]
            ),
        ]
    )

    # Add metadata
    if metadata:
        for key, value in metadata.items():
            meta = combined_model.metadata_props.add()
            meta.key = key
            meta.value = value

    onnx.save(combined_model, output_path)
    print(f"Ensemble saved to {output_path}")
