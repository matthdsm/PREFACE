import unittest
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from xgboost import XGBRegressor

from preface.lib.xgboost import xgboost_fit, xgboost_tune
from preface.lib.impute import ImputeOptions


class TestXGBoost(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_xgboost_outputs")
        self.test_dir.mkdir(exist_ok=True)

        # Create dummy data
        self.n_samples = 50
        self.n_features = 5
        self.x = np.random.rand(self.n_samples, self.n_features)
        self.y = np.random.rand(self.n_samples)
        self.groups = np.random.randint(0, 5, self.n_samples)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_xgboost_fit(self):
        params = {"n_estimators": 10, "max_depth": 2}

        # Split data
        x_train, x_test = self.x[:40], self.x[40:]
        y_train, y_test = self.y[:40], self.y[40:]

        model, predictions = xgboost_fit(x_train, x_test, y_train, y_test, params)

        self.assertIsInstance(model, XGBRegressor)
        self.assertEqual(predictions.shape, (10,))
        self.assertTrue(hasattr(model, "feature_importances_"))

    @patch("preface.lib.xgboost.optuna")
    def test_xgboost_tune(self, mock_optuna):
        # Mock study and optimization
        mock_study = MagicMock()
        mock_study.best_params = {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        mock_optuna.create_study.return_value = mock_study

        # Run tuning
        best_params = xgboost_tune(
            self.x,
            self.y,
            self.groups,
            n_components=3,
            outdir=self.test_dir,
            impute_option=ImputeOptions.ZERO,
        )

        self.assertEqual(best_params, mock_study.best_params)
        mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
