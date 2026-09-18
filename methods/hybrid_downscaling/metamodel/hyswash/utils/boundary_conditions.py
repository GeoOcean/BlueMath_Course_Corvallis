
import numpy as np
from bluemath_tk.waves.series import waves_dispersion

def get_real_scenarios_from_dataset(lhs_dataset, h0):
    """
    Function to filter the LHS dataset to obtain real scenarios.
    It filters out scenarios that are not physically realistic based on the wave dispersion relation.

    Parameters
    ----------
    lhs_dataset : pandas.DataFrame
        DataFrame containing the Latin Hypercube Sampling data with columns:
        'Hs', 'Hs_L0', 'Wv', 'hv', 'Nv'

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
    df_centroids = df_centroids.loc[(df_centroids["kh"] < 1) & (df_centroids["Tp"] > 7)]
    df_centroids = df_centroids.loc[(df_centroids["h/L"] < 0.5)]
    df_dataset = lhs_dataset.loc[df_centroids.index]

    return df_dataset