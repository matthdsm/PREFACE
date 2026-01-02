import unittest
import numpy as np
import onnx
from sklearn.decomposition import PCA
from preface.lib.functions import pca_export


class TestPcaExport(unittest.TestCase):
    def test_pca_export(self):
        # 1. Create dummy data and fit a PCA model
        n_samples = 100
        n_features = 20
        n_components = 5
        data = np.random.rand(n_samples, n_features).astype(np.float32)

        pca = PCA(n_components=n_components)
        pca.fit(data)

        # 2. Export to ONNX
        onnx_model = pca_export(pca, n_features)

        # 3. Verify
        self.assertIsInstance(onnx_model, onnx.ModelProto)

        # Check graph properties
        self.assertGreater(len(onnx_model.graph.node), 0)
        self.assertEqual(len(onnx_model.graph.input), 1)
        self.assertEqual(onnx_model.graph.input[0].name, "input")
        self.assertEqual(len(onnx_model.graph.output), 1)

        # Validate the model
        onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    unittest.main()
