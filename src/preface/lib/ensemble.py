from pathlib import Path

import numpy as np
import onnx
import onnxmltools
import tensorflow as tf
import tf2onnx
from onnx import TensorProto, helper
from onnx.compose import add_prefix, merge_models
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
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
    models: list[tuple[object, PCA, Model | XGBRegressor | dict[str, SVR | SVC]]],
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
    fold_info = []  # Store (model_type, reg_base, class_base) for each fold

    # Process each fold
    for i, (imputer, pca, model) in enumerate(models):
        fold_prefix = f"fold_{i}_"

        # Convert Imputer
        initial_type = [("input", FloatTensorType([None, input_dim]))]

        # sklearn imputer
        imputer_onnx = convert_sklearn(
            imputer, initial_types=initial_type, target_opset=12
        )
        _ensure_opset(imputer_onnx, 12)

        # Prefix Imputer
        imputer_onnx = add_prefix(imputer_onnx, prefix="imputer_")  # type: ignore
        imputer_out_name = imputer_onnx.graph.output[0].name  # type: ignore

        # Convert PCA
        # Imputer -> PCA
        # Use pca.n_features_in_ to handle case where imputer reduced dimensions
        n_features_pca = getattr(pca, "n_features_in_", input_dim)
        pca_initial_type = [("input_pca", FloatTensorType([None, n_features_pca]))]
        pca_onnx = convert_sklearn(pca, initial_types=pca_initial_type, target_opset=12)
        _ensure_opset(pca_onnx, 12)

        # Prefix PCA
        pca_onnx = add_prefix(pca_onnx, prefix="pca_")  # type: ignore
        pca_in_name = pca_onnx.graph.input[0].name  # type: ignore
        pca_out_name = pca_onnx.graph.output[0].name  # type: ignore

        # Merge Imputer + PCA
        current_model = merge_models(
            imputer_onnx,  # type: ignore
            pca_onnx,  # type: ignore
            io_map=[(imputer_out_name, pca_in_name)],  # type: ignore
        )
        # Update output name
        current_out_name = pca_out_name

        # 3. Convert Model
        # Input to Model is PCA output. Shape: [None, n_components]
        n_comps = pca.n_components_

        model_type = "unknown"
        if isinstance(model, Model):
            model_type = "nn"
        elif isinstance(model, (dict, list, tuple)):  # Handle dict for SVM
            model_type = "svm"
        else:
            model_type = "xgb"

        if model_type == "nn":
            spec = (tf.TensorSpec((None, n_comps), tf.float32, name="input_model"),)  # type: ignore
            m_onnx, _ = tf2onnx.convert.from_keras(
                model, input_signature=spec, opset=12
            )
            m_onnx.graph.name = f"fold_{i}_model"

            # Prefix NN
            m_onnx = add_prefix(m_onnx, prefix="nn_")
            m_in_name = m_onnx.graph.input[0].name

            # Merge (Imputer+PCA) + Model
            current_model = merge_models(
                current_model,  # type: ignore
                m_onnx,
                io_map=[(current_out_name, m_in_name)],  # type: ignore
            )

            # Record outputs
            # Assuming standard naming for NN for now as we can't easily inspect without graph structure knowledge
            fold_info.append(("nn", "nn_reg_output", "nn_class_output"))

        elif model_type == "xgb":
            m_onnx = onnxmltools.convert_xgboost(
                model,
                initial_types=[("input_model", FloatTensorType([None, n_comps]))],
                target_opset=12,
            )

            # Prefix XGB
            m_onnx = add_prefix(m_onnx, prefix="xgb_")
            m_in_name = m_onnx.graph.input[0].name

            # Capture output name
            xgb_out_base = m_onnx.graph.output[0].name

            # Merge (Imputer+PCA) + Model
            current_model = merge_models(
                current_model,  # type: ignore
                m_onnx,
                io_map=[(current_out_name, m_in_name)],  # type: ignore
            )

            fold_info.append(("xgb", xgb_out_base, None))

        elif model_type == "svm":
            # Expecting dict with 'SVR' and 'SVC'
            svr = model["SVR"]
            svc = model["SVC"]

            # Patch SVR if no support vectors (e.g. large epsilon)
            if hasattr(svr, "support_vectors_") and svr.support_vectors_.shape[0] == 0:
                # Add dummy support vector with 0 weight
                dummy_sv = np.zeros(
                    (1, svr.support_vectors_.shape[1]), dtype=np.float32
                )
                svr.support_vectors_ = dummy_sv

                # Patch internal attributes used by coef_ property
                svr._dual_coef_ = np.zeros((1, 1), dtype=np.float32)
                svr.dual_coef_ = svr._dual_coef_

                if hasattr(svr, "_n_support"):
                    svr._n_support = np.array([1], dtype=np.int32)

            # Convert SVR
            svr_onnx = convert_sklearn(
                svr,
                initial_types=[("input_svr", FloatTensorType([None, n_comps]))],
                target_opset=12,
            )
            _ensure_opset(svr_onnx, 12)

            # Prefix SVR
            svr_onnx = add_prefix(svr_onnx, prefix="svr_")
            svr_in_name = svr_onnx.graph.input[0].name
            svr_out_base = svr_onnx.graph.output[0].name

            # Convert SVC
            # zipmap=False is important to get probabilities as tensor
            svc_onnx = convert_sklearn(
                svc,
                initial_types=[("input_svc", FloatTensorType([None, n_comps]))],
                target_opset=12,
                options={"zipmap": False},
            )
            _ensure_opset(svc_onnx, 12)

            # Prefix SVC
            svc_onnx = add_prefix(svc_onnx, prefix="svc_")
            svc_in_name = svc_onnx.graph.input[0].name
            # Output 0 is label, Output 1 is probabilities (usually)
            svc_out_base = svc_onnx.graph.output[1].name

            # Combine SVR and SVC into one model (parallel branches)
            svm_combined = merge_models(svr_onnx, svc_onnx, io_map=[])

            # Merge (Imputer+PCA) with (SVR+SVC)
            # Connect PCA output to both SVR and SVC inputs
            current_model = merge_models(
                current_model,
                svm_combined,
                io_map=[
                    (current_out_name, svr_in_name),
                    (current_out_name, svc_in_name),
                ],
            )

            fold_info.append(("svm", svr_out_base, svc_out_base))

        # Prefix everything in this fold's graph
        prefixed_model = add_prefix(current_model, prefix=fold_prefix)
        prefixed_models.append(prefixed_model)

    # --- Merge All Folds ---
    # We want a single input "input" that feeds into all fold_i_input

    # Start with the first fold
    combined_model = prefixed_models[0]

    for i in range(1, len(prefixed_models)):
        combined_model = merge_models(
            combined_model,
            prefixed_models[i],
            io_map=[],
        )

    graph = combined_model.graph

    # Find all inputs that look like "fold_X_input" or "fold_X_input_pca"
    fold_inputs = []
    for node in graph.input:
        if node.name.endswith("input") or node.name.endswith("input_pca"):
            fold_inputs.append(node.name)

    # Create a global input
    global_input_name = "input"

    # Replace all usages of "fold_i_input" with "global_input".
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
    reg_names = []
    class_names = []

    # Helper to find output names
    for i, (model_type, reg_base, class_base) in enumerate(fold_info):
        prefix = f"fold_{i}_"

        if model_type == "nn":
            # Keras outputs are typically named by layer names.
            # We added "nn_" prefix.
            reg_names.append(prefix + reg_base)
            class_names.append(prefix + class_base)

        elif model_type == "xgb":
            # XGBoost output "variable" -> "xgb_variable"
            xgb_out = prefix + reg_base

            # We need to split this output. It is [Batch, 2].
            # Column 0: Reg, Column 1: Class (Prob)

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
            # SVR output "variable" -> "svr_variable"
            # SVC output "output_probability" -> "svc_output_probability"

            svm_reg_out = prefix + reg_base
            svm_class_prob_out = prefix + class_base

            reg_names.append(svm_reg_out)

            # SVC probability output is [Batch, 2] (prob class 0, prob class 1).
            # We need to extract the second column (class 1).

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
