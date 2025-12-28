import logging
from enum import Enum

import numpy as np
import numpy.typing as npt
import sklearn
from sklearn.experimental import enable_iterative_imputer  # pylint: disable=unused-import # noqa: F401  # type: ignore
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


class ImputeOptions(Enum):
    ZERO = "zero"  # assume missing values are zero
    MICE = "mice"  # impute missing values using MICE
    MEAN = "mean"  # impute missing values by calculating mean
    MEDIAN = "median"  # impute missing values by calculating median
    KNN = "knn"  # impute missing values using k-nearest neighbors


def impute_nan(
    values: npt.NDArray, method: ImputeOptions
) -> tuple[npt.NDArray, object]:
    """
    Handle NaN values
    Identify the type of missingness (MCAR, MAR, MNAR). Here we assume MAR.
    Here the input log2 ratios indicate relative coverage to a reference,
    we can either impute missing values or assume zero (no change).

    Returns:
        tuple: (imputed_values, fitted_imputer)
        Note: For ImputeOptions.ZERO, the fitted_imputer is None (handled manually).
    """
    imputer = None

    # Option 1: Impute NaN through MICE (Multiple Imputation by Chained Equations)
    if method == ImputeOptions.MICE:
        # Check sklearn version for compatibility
        sk_version = sklearn.__version__
        if sk_version != "1.8.0":
            logging.warning(f"""PREFACE uses imputation and was developed using scikit-learn version 1.8.0.
                        Since imputation is still experimental, it may be subject to change in other versions.
                        You are using version {sk_version}. Proceed with caution.""")

        logging.info("Imputing missing values using MICE... This might take a while.")
        imputer = IterativeImputer(
            random_state=42, max_iter=10, initial_strategy="mean"
        )
        imputed_values = imputer.fit_transform(values)

    # Option 2: Assume missing values are zero (no change)
    elif method == ImputeOptions.ZERO:
        logging.info("Assuming missing values are zero...")
        imputed_values = np.where(np.isnan(values), 0.0, values)
        # For ZERO, we don't have a sklearn imputer, but we can simulate one or handle it in export
        imputer = None

    # Option 3: Impute missing values by calculating mean
    elif method == ImputeOptions.MEAN:
        logging.info("Imputing missing values using mean strategy...")
        imputer = SimpleImputer(strategy="mean")
        imputed_values = imputer.fit_transform(values)

    # Option 4: Impute missing values by calculating median
    elif method == ImputeOptions.MEDIAN:
        logging.info("Imputing missing values using median strategy...")
        imputer = SimpleImputer(strategy="median")
        imputed_values = imputer.fit_transform(values)

    # Option 5: Impute missing values using k-nearest neighbors
    elif method == ImputeOptions.KNN:
        logging.info("Imputing missing values using k-nearest neighbors...")
        imputer = KNNImputer(n_neighbors=5)
        imputed_values = imputer.fit_transform(values)

    return imputed_values, imputer
