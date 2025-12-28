from pathlib import Path

import onnx
import onnxmltools
from onnx import TensorProto, ModelProto, GraphProto, FunctionProto, helper
from onnx.compose import add_prefix, merge_models
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA
from tensorflow.keras import Model  # type: ignore
from xgboost import XGBRegressor


def build_ensemble(
    pca_obj: PCA, models: list[Model | XGBRegressor], input_dim: int, output_path: Path
) -> tuple[ModelProto | GraphProto | FunctionProto]:
    """
    Save an ensemble of models (either Keras NNs or XGBoost regressors) combined with a PCA
    """

    initial_type = [("input", FloatTensorType([None, input_dim]))]
    pca_onnx = convert_sklearn(pca_obj, initial_types=initial_type, target_opset=12)
    pca_out_name = pca_onnx.graph.output[0].name  # type: ignore

    prefixed_models = []
    model_type = "nn" if isinstance(models[0], Model) else "xgb"
    for i, m in enumerate(models):
        if model_type == "nn":
            m_onnx = onnxmltools.convert_keras(m, name=f"fold_{i}")
        elif model_type == "xgb":
            m_onnx = onnxmltools.convert_xgboost(
                m,
                initial_types=[
                    (pca_out_name, FloatTensorType([None, pca_obj.n_components_]))
                ],
            )
        else:
            raise ValueError("Model must be either Keras Model or XGBRegressor")

        # Prefixing prevents node name collisions between the 10 folds
        prefixed_models.append(add_prefix(m_onnx, prefix=f"fold_{i}_"))

    # --- 2. Merge Graphs ---
    combined_model = pca_onnx
    for i in range(len(prefixed_models)):
        combined_model = merge_models(
            combined_model,  # type: ignore
            prefixed_models[i],
            io_map=[(pca_out_name, f"fold_{i}_{pca_out_name}")],
        )

    graph = combined_model.graph  # type: ignore

    # --- 3. Identify Output Names for Averaging ---
    # For NN: Keras usually names outputs after the final layer (e.g., 'reg_out', 'class_out')
    # For XGB: Multi-output trees usually output a single tensor that we must split
    if model_type == "nn":
        reg_names = [f"fold_{i}_reg_output" for i in range(len(models))]
        class_names = [f"fold_{i}_class_output" for i in range(len(models))]
    else:
        # XGB strategy: We split the combined output tensor [Batch, 2] into two
        reg_names, class_names = [], []
        for i in range(len(models)):
            reg_node_out = f"fold_{i}_reg_split"
            class_node_out = f"fold_{i}_class_split"
            reg_names.append(reg_node_out)
            class_names.append(class_node_out)

    # --- 4. Add Mean Nodes for both heads ---
    final_reg_name = "final_ff_score"
    final_class_name = "final_sex_prob"

    mean_reg_node = helper.make_node(
        "Mean", inputs=reg_names, outputs=[final_reg_name], name="Mean_FF"
    )
    mean_class_node = helper.make_node(
        "Mean", inputs=class_names, outputs=[final_class_name], name="Mean_Sex"
    )

    graph.node.extend([mean_reg_node, mean_class_node])

    # --- 5. Clean up and finalize outputs ---
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

    onnx.save(combined_model, output_path)  # type: ignore
    print(f"Dual-head ensemble saved to {output_path}")

    return combined_model  # type: ignore
