import numpy as np
import pandas as pd
import xarray as xr


def pcs_to_flood_maps(pca, pcs, var="zsmax"):
    """
    Reconstruct flood depth maps from principal components.

    Parameters
    ----------
    pca : PCA
        Fitted bluemath_tk PCA model.
    pcs : pd.DataFrame
        Principal components, one row per scenario (columns PC1, PC2, ...).
        Extra columns (e.g. confidence intervals) are ignored.
    var : str, optional
        Name of the reconstructed variable.

    Returns
    -------
    xr.DataArray
        Flood depth maps with dimensions (case_num, y, x). Negative
        depths produced by the reconstruction are set to zero.
    """
    pc_names = list(pca.pcs_df.columns)

    pcs_xr = xr.DataArray(
        pcs[pc_names].values,
        dims=[pca.pca_dim_for_rows, "PCs"],
        coords={pca.pca_dim_for_rows: pcs.index.values, "PCs": pc_names},
    )

    flood_maps = pca.inverse_transform(pcs_xr)[var].clip(min=0)

    return flood_maps.transpose(pca.pca_dim_for_rows, "y", "x")


def sample_pcs(pcs, pc_names, n_samples=100, seed=0):
    """
    Draw random PC samples from the GP predictive distribution.

    The GP returns, for each PC, the mean and the confidence region
    (mean ± 2 std). Each PC is sampled independently from a normal
    distribution with that mean and standard deviation.

    Parameters
    ----------
    pcs : pd.Series
        GP prediction for a single scenario (PC means and *_lower_ci /
        *_upper_ci columns).
    pc_names : list[str]
        Names of the PCs.
    n_samples : int, optional
        Number of samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Sampled PCs with shape (n_samples, n_pcs).
    """
    rng = np.random.default_rng(seed)

    mean = pcs[pc_names].values.astype(float)
    std = np.array(
        [(pcs[f"{pc}_upper_ci"] - pcs[f"{pc}_lower_ci"]) / 4 for pc in pc_names],
        dtype=float,
    )

    return pd.DataFrame(
        rng.normal(mean, std, size=(n_samples, len(pc_names))),
        columns=pc_names,
    )


def flooded_area_km2(flood_maps, threshold=0.1):
    """
    Flooded area [km2], counting the pixels with a flood depth above `threshold` [m].
    """
    pixel_area = abs(
        float(flood_maps.x[1] - flood_maps.x[0]) * float(flood_maps.y[1] - flood_maps.y[0])
    )

    return (flood_maps > threshold).sum(dim=["x", "y"]) * pixel_area / 1e6
