
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

def reconstruct_binwaves_point(offshore_spectra, kp_coeffs, swan_cases, lon, lat):
    """
    Reconstruct the wave climate at the BinWaves output site closest to (lon, lat).

    It follows the BinWaves reconstruction: the offshore spectrum is smoothed
    and coarsened, its energy in each case bin (freq, dir) is weighted by the
    propagation coefficients (kp) of the site, and all cases are added up.

    Parameters
    ----------
    offshore_spectra : xarray.Dataset
        Offshore spectra with "efth" (time, freq, dir) in m2/Hz/deg, as saved
        by the NDBC notebook.
    kp_coeffs : xarray.Dataset
        BinWaves propagation coefficients "kps" (case_num, site, freq, dir),
        with site coordinates "coord_x" and "coord_y".
    swan_cases : pandas.DataFrame
        BinWaves cases with the "freq" and "dir" of each case_num.
    lon, lat : float
        Location of interest.

    Returns
    -------
    xarray.Dataset
        Hs, Tp and Hs_L0 time series at the selected site, which is stored in
        the attributes with its coordinates.
    """
    import wavespectra  # noqa: F401 (registers the .spec accessor)
    import xarray as xr

    site = int(
        np.argmin(
            np.hypot(kp_coeffs.coord_x.values - lon, kp_coeffs.coord_y.values - lat)
        )
    )
    kp = kp_coeffs["kps"].isel(site=site)

    # Same preprocessing as in BinWaves (smoothing in blocks of time to save memory)
    efth = offshore_spectra["efth"]
    efth = xr.concat(
        [
            efth.isel(time=slice(i, i + 500))
            .rolling(freq=5, center=True, min_periods=1)
            .mean()
            for i in range(0, efth.sizes["time"], 500)
        ],
        dim="time",
    )
    efth = efth.coarsen(freq=3, dir=3, boundary="pad").mean()
    efth = efth.reindex(freq=kp.freq, dir=kp.dir, method="nearest")

    # Offshore energy in the bin of each case
    cases = swan_cases.loc[kp.case_num.values]
    case_energy = efth.sel(
        freq=xr.DataArray(cases["freq"].values, dims="case_num"),
        dir=xr.DataArray(cases["dir"].values, dims="case_num"),
        method="nearest",
    ).drop_vars(["freq", "dir"]).assign_coords(case_num=kp.case_num.values)

    # Onshore spectrum: weighted sum of the propagated cases
    onshore = xr.dot(case_energy, kp, dim="case_num").rename("efth")

    hs = onshore.spec.hs().reset_coords(drop=True)
    tp = onshore.spec.tp().reset_coords(drop=True)
    return xr.Dataset(
        {"Hs": hs, "Tp": tp, "Hs_L0": hs / (9.806 * tp**2 / (2 * np.pi))},
        attrs={
            "site": site,
            "lon": float(kp_coeffs.coord_x[site]),
            "lat": float(kp_coeffs.coord_y[site]),
        },
    ).dropna("time")
