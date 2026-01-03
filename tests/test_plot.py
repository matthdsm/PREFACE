"""Unit tests for plotting functions."""

import shutil
import unittest
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

# Import the functions to be tested, including the private one for direct testing
from preface.lib.plot import (
    _calculate_regression_metrics,
    plot_ffx,
    plot_pca,
    plot_regression_performance,
    plot_tsne,
    plot_cv_splits,
)
from sklearn.model_selection import GroupShuffleSplit


class TestPlottingFunctions(unittest.TestCase):
    """Test case for the plotting and RLM fitting functions."""

    def setUp(self):
        """Set up a temporary directory for plot outputs."""
        self.test_dir = Path("test_plot_outputs")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def test_calculate_regression_metrics(self):
        """Test the private _calculate_regression_metrics function."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.3, 5.0])
        metrics = _calculate_regression_metrics(y_true, y_pred)

        self.assertIn("mae", metrics)
        self.assertIn("slope", metrics)
        self.assertIn("intercept", metrics)
        self.assertIn("correlation", metrics)
        self.assertAlmostEqual(metrics["mae"], 0.14, places=2)
        self.assertAlmostEqual(metrics["slope"], 1.0, places=2)

    def test_plot_regression_performance_smoke(self):
        """Smoke test for plot_regression_performance."""
        y_true = np.random.rand(50) * 20
        y_pred = y_true + np.random.randn(50)
        pca_variance = np.array([0.5, 0.2, 0.1, 0.05, 0.02])
        output_path = self.test_dir / "regression_performance.png"

        metrics = plot_regression_performance(
            y_pred,
            y_true,
            pca_variance,
            n_feat=3,
            xlab="Predicted",
            ylab="True",
            path=output_path,
        )
        self.assertTrue(output_path.exists())
        self.assertIn("mae", metrics)

    def test_plot_ffx_smoke(self):
        """Smoke test for plot_ffx."""
        x = np.random.rand(50) * 20
        y = 0.5 * x + 2 + np.random.randn(50)
        output_path = self.test_dir / "ffx_plot.png"
        plot_ffx(x, y, intercept=2.0, slope=0.5, output=output_path)
        self.assertTrue(output_path.exists())

    def test_plot_pca_smoke(self):
        """Smoke test for plot_pca."""
        data = np.random.rand(50, 10)
        pca = PCA(n_components=2).fit(data)
        components = pca.transform(data)
        labels = ["A"] * 25 + ["B"] * 25
        output_path = self.test_dir / "pca_plot.png"

        # Test with labels
        plot_pca(pca, components, output_path, labels=labels)
        self.assertTrue(output_path.exists())

        # Test without labels
        plot_pca(pca, components, output_path)
        self.assertTrue(output_path.exists())

    def test_plot_tsne_smoke(self):
        """Smoke test for plot_tsne."""
        data = np.random.rand(30, 10)  # Perplexity requires n_samples > perplexity
        labels = ["A"] * 15 + ["B"] * 15
        output_path = self.test_dir / "tsne_plot.png"

        # Test with labels
        plot_tsne(data, output_path, labels=labels, perplexity=10)
        self.assertTrue(output_path.exists())

        # Test without labels
        plot_tsne(data, output_path, perplexity=10)
        self.assertTrue(output_path.exists())

    def test_plot_cv_splits_smoke(self):
        """Smoke test for plot_cv_splits."""
        X = np.random.rand(20, 2)
        y = np.random.rand(20)
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
        cv = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
        output_path = self.test_dir / "cv_splits.png"

        plot_cv_splits(cv, X, y, groups, output_path)
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
