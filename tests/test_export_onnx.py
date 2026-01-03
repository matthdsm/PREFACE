import unittest
import shutil
import numpy as np
import onnx
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from xgboost import XGBRegressor

from preface.lib.functions import pca_export, ensemble_export


class TestExportOnnx(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_onnx_outputs")
        self.test_dir.mkdir(exist_ok=True)
        self.input_dim = 10

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_pca_export(self):
        # Create and fit PCA
        x = np.random.rand(20, self.input_dim)
        pca = PCA(n_components=5)
        pca.fit(x)

        # Export
        onnx_model = pca_export(pca, self.input_dim)

        self.assertIsInstance(onnx_model, onnx.ModelProto)
        onnx.checker.check_model(onnx_model)

        # Check input/output
        self.assertEqual(len(onnx_model.graph.input), 1)
        self.assertEqual(onnx_model.graph.input[0].name, "pca_input")

    def test_ensemble_export(self):
        output_path = self.test_dir / "ensemble.onnx"

        # Create dummy models for 2 splits
        models = []
        for _ in range(2):
            imputer = SimpleImputer(strategy="mean")
            imputer.fit(np.random.rand(10, self.input_dim))

            pca = PCA(n_components=5)
            pca.fit(np.random.rand(10, self.input_dim))

            # Use SVR as it is simpler to setup without huge deps
            svr = SVR(kernel="linear")
            svr.fit(np.random.rand(10, 5), np.random.rand(10))

            models.append((imputer, pca, svr))

        # Export ensemble
        ensemble_export(
            models,
            input_dim=self.input_dim,
            output_path=output_path,
            metadata={"version": "1.0"},
        )

        self.assertTrue(output_path.exists())

        # Load and verify
        model = onnx.load(output_path)
        onnx.checker.check_model(model)

        # Check metadata
        meta_dict = {p.key: p.value for p in model.metadata_props}
        self.assertEqual(meta_dict.get("version"), "1.0")


if __name__ == "__main__":
    unittest.main()
