import numpy.typing as npt
import onnx
import pandas as pd
import statsmodels.api as sm
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA


def preprocess_ratios(ratios_df: pd.DataFrame, exclude_chrs: list[str]) -> pd.DataFrame:
    """Preprocess ratios DataFrame by excluding chromosomes, adding region column and transposing.
    returns a x by 1 dataframe with regions as columns.
    """
    # sanitize columns
    ratios_df = ratios_df[["chr", "start", "end", "ratio"]].copy()
    # santize chr column
    ratios_df["chr"] = ratios_df["chr"].astype(str).str.replace("chr", "", regex=False)
    # exclude chromosomes
    ratios_df = ratios_df[~ratios_df["chr"].isin(exclude_chrs)].copy()
    # add region column
    ratios_df["region"] = (
        ratios_df["chr"]
        + ":"  # type: ignore
        + ratios_df["start"].astype(str)
        + "-"
        + ratios_df["end"].astype(str)
    )  # type: ignore
    # drop chr, start, end columns
    ratios_df.drop(columns=["chr", "start", "end"], inplace=True)
    # set region as index and transpose
    ratios_df = ratios_df.set_index("region").T

    return ratios_df


def fit_rlm(x_values: npt.NDArray, y_values: npt.NDArray) -> tuple[float, float]:
    """
    Fit robust linear model (RLM) and return intercept and slope.
    """

    x_rlm = sm.add_constant(x_values)
    rlm_model = sm.RLM(y_values, x_rlm, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    fit_params = rlm_results.params
    intercept: float = fit_params[0]
    slope: float = fit_params[1]
    return intercept, slope


def pca_export(pca: PCA, input_dim: int) -> onnx.ModelProto:
    """Export PCA model to ONNX format."""
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    pca_onnx = convert_sklearn(pca, initial_types=initial_type, target_opset=18)
    return pca_onnx  # type: ignore
