import unittest
import numpy as np
import onnx
from preface.lib.svm import svm_export, svm_fit


class TestSvmExport(unittest.TestCase):
    def test_svm_export(self):
        # 1. Create dummy data
        n_samples = 100
        n_features = 10
        x_train = np.random.rand(n_samples, n_features).astype(np.float32)
        y_train = np.random.rand(n_samples, 1).astype(np.float32)
        x_test = np.random.rand(n_samples, n_features).astype(np.float32)
        y_test = np.random.rand(n_samples, 1).astype(np.float32)

        # 2. Train a model using svm_fit
        model, _ = svm_fit(x_train, x_test, y_train, y_test, params={})

        # 3. Export to ONNX
        onnx_model = svm_export(model)

        # 4. Verify
        self.assertIsInstance(onnx_model, onnx.ModelProto)

        # Check graph properties
        self.assertGreater(len(onnx_model.graph.node), 0)
        self.assertEqual(len(onnx_model.graph.input), 1)
        self.assertEqual(len(onnx_model.graph.output), 1)
        self.assertEqual(onnx_model.graph.input[0].name, "svm_input")

        # Validate the model
        onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    unittest.main()
