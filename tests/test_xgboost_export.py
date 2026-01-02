import unittest
import numpy as np
import onnx
from xgboost import XGBRegressor
from preface.lib.xgboost import xgboost_export


class TestXGBoostExport(unittest.TestCase):
    def test_xgboost_export(self):
        # 1. Create dummy data and train a model
        n_samples = 100
        n_features = 10
        x_train = np.random.rand(n_samples, n_features).astype(np.float32)
        y_train = np.random.rand(n_samples).astype(np.float32)
        
        model = XGBRegressor(n_estimators=3, max_depth=3, base_score=0.5, random_state=42)
        model.fit(x_train, y_train)
        
        # 2. Export to ONNX
        onnx_model = xgboost_export(model)

        # 3. Verify
        self.assertIsInstance(onnx_model, onnx.ModelProto)
        
        # Check graph properties
        self.assertGreater(len(onnx_model.graph.node), 0)
        self.assertEqual(len(onnx_model.graph.input), 1)
        self.assertEqual(onnx_model.graph.input[0].name, "input")
        
        # Validate the model
        onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
    unittest.main()
