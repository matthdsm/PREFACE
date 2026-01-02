import unittest
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from preface.lib.impute import impute_nan, ImputeOptions


class TestImpute(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with known properties
        # shape (5, 3)
        self.data = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [np.nan, np.nan, np.nan],
            ]
        )
        # Add some NaNs in other places
        self.data[0, 0] = np.nan  # was 1.0
        self.data[2, 1] = np.nan  # was 8.0

        # Create a version with columns that have clear means/medians
        # Col 0: [?, 4, 7, 10, ?] -> valid: 4, 7, 10. Mean=7, Median=7
        # Col 1: [2, 5, ?, 11, ?] -> valid: 2, 5, 11. Mean=6, Median=5
        # Col 2: [3, 6, 9, 12, ?] -> valid: 3, 6, 9, 12. Mean=7.5, Median=7.5

        # Refined data for checking values
        self.data_check = np.array(
            [
                [np.nan, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, np.nan, 9.0],
                [10.0, 11.0, 12.0],
                [np.nan, np.nan, np.nan],
            ]
        )

        # NOTE: SimpleImputer with default axis=0 (columns)
        # Col 0 valid: 4, 7, 10 -> Mean: 7.0, Median: 7.0
        # Col 1 valid: 2, 5, 11 -> Mean: 6.0, Median: 5.0
        # Col 2 valid: 3, 6, 9, 12 -> Mean: 7.5, Median: 7.5 (avg of 6 and 9)
        # Note: Median of [3,6,9,12] is (6+9)/2 = 7.5

    def test_impute_zero(self):
        """Test ZERO imputation."""
        imputed_data, imputer = impute_nan(self.data_check.copy(), ImputeOptions.ZERO)

        self.assertIsInstance(imputer, SimpleImputer)
        self.assertFalse(np.isnan(imputed_data).any())

        # Check specific values were replaced by 0
        self.assertEqual(imputed_data[0, 0], 0.0)
        self.assertEqual(imputed_data[2, 1], 0.0)
        self.assertEqual(imputed_data[4, 0], 0.0)

        # Check non-NaN values are preserved
        self.assertEqual(imputed_data[1, 0], 4.0)

    def test_impute_mean(self):
        """Test MEAN imputation."""
        imputed_data, imputer = impute_nan(self.data_check.copy(), ImputeOptions.MEAN)

        self.assertIsInstance(imputer, SimpleImputer)
        self.assertFalse(np.isnan(imputed_data).any())

        # Col 0 mean is 7.0
        self.assertAlmostEqual(imputed_data[0, 0], 7.0)
        # Col 1 mean is 6.0
        self.assertAlmostEqual(imputed_data[2, 1], 6.0)

    def test_impute_median(self):
        """Test MEDIAN imputation."""
        imputed_data, imputer = impute_nan(self.data_check.copy(), ImputeOptions.MEDIAN)

        self.assertIsInstance(imputer, SimpleImputer)
        self.assertFalse(np.isnan(imputed_data).any())

        # Col 0 median is 7.0
        self.assertAlmostEqual(imputed_data[0, 0], 7.0)
        # Col 1 median is 5.0
        self.assertAlmostEqual(imputed_data[2, 1], 5.0)

    def test_impute_knn(self):
        """Test KNN imputation."""
        # Need enough samples for neighbors, defaulting to 5 in implementation
        # But our data is small (5 rows). KNNImputer(n_neighbors=5) might be capped by samples.
        # The implementation uses default n_neighbors=5.
        # With 5 samples, it will use available neighbors.

        imputed_data, imputer = impute_nan(self.data_check.copy(), ImputeOptions.KNN)

        self.assertIsInstance(imputer, KNNImputer)
        self.assertFalse(np.isnan(imputed_data).any())
        self.assertEqual(imputed_data.shape, self.data_check.shape)

    def test_impute_mice(self):
        """Test MICE imputation."""
        # Suppress logging for clean test output or check it if needed
        # Just checking it runs and returns IterativeImputer

        imputed_data, imputer = impute_nan(self.data_check.copy(), ImputeOptions.MICE)

        self.assertIsInstance(imputer, IterativeImputer)
        self.assertFalse(np.isnan(imputed_data).any())
        self.assertEqual(imputed_data.shape, self.data_check.shape)


if __name__ == "__main__":
    unittest.main()
