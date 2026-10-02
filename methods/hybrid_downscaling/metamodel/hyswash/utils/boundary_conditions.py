
import numpy as np
from bluemath_tk.waves.series import waves_dispersion

def get_real_scenarios_from_dataset(lhs_dataset, h0, tp_min=7, tp_max=None):
    """
    Function to filter the LHS dataset to obtain real scenarios.
    It filters out scenarios that are not physically realistic based on the wave dispersion relation.

    Parameters
    ----------
    lhs_dataset : pandas.DataFrame
        DataFrame containing the Latin Hypercube Sampling data with columns:
        'Hs', 'Hs_L0', 'Wv', 'hv', 'Nv'
    h0 : float
        Water depth at the offshore boundary of the profile (m).
    tp_min, tp_max : float, optional
        Range of peak periods (s) to keep. tp_max=None means no upper limit.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only physically realistic scenarios.
    """

    # Calculation of the wave length and other parameters to avoid unphysical values
    df_centroids = lhs_dataset.copy()
    df_centroids["Tp"] = np.sqrt(
        (df_centroids["Hs"].values * 2 * np.pi) / (9.806 * df_centroids["Hs_L0"])
    )
    df_centroids["L"] = [waves_dispersion(i, h0)[0] for i in df_centroids["Tp"]]
    df_centroids["h/L"] = h0 / df_centroids["L"]
    df_centroids["kh"] = (2 * np.pi / df_centroids["L"]) * h0
    # SWASH with one vertical layer and the box scheme (VERT 1, NONHYD BOX)
    # represents dispersion well up to kh ~ 3; Tp > 7 s keeps kh < ~2.5 at 30 m
    df_centroids = df_centroids.loc[(df_centroids["kh"] < 3) & (df_centroids["Tp"] > tp_min)]
    if tp_max is not None:
        df_centroids = df_centroids.loc[df_centroids["Tp"] < tp_max]
    df_centroids = df_centroids.loc[(df_centroids["h/L"] < 0.5)]
    df_dataset = lhs_dataset.loc[df_centroids.index]

    return df_dataset