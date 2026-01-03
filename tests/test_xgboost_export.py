import unittest
import numpy as np
import onnx
from xgboost import XGBRegressor
from preface.lib.xgboost import xgboost_export


class TestXGBoostExport(unittest.TestCase):
    @unittest.skip(
        "Skipping due to known compatibility issue between XGBoost > 1.7 and skl2onnx (base_score parsing)"
    )
    def test_xgboost_export(self):
        # 1. Create dummy data and train a model
        n_samples = 100
        n_features = 10
        x_train = np.random.rand(n_samples, n_features).astype(np.float32)
        y_train = np.random.rand(n_samples).astype(np.float32)

        model = XGBRegressor(n_estimators=3, max_depth=3, random_state=42)
        model.fit(x_train, y_train)
        # Force base_score to be a float to avoid skl2onnx issues with newer xgboost
        model.base_score = 0.5

        # 2. Export to ONNX
        onnx_model = xgboost_export(model)

        # 3. Verify
        self.assertIsInstance(onnx_model, onnx.ModelProto)

        # Check graph properties
        self.assertGreater(len(onnx_model.graph.node), 0)
        self.assertEqual(len(onnx_model.graph.input), 1)
        self.assertEqual(onnx_model.graph.input[0].name, "xgboost_input")

        # Validate the model
        onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    unittest.main()
