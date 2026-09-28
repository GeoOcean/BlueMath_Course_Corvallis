import os
import re
import shutil
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr


def get_available_years(buoy_id: str, subdir: str, suffix: str) -> List[int]:
    """
    Find which years have historical data for a buoy by parsing NDBC's
    station history page (e.g. subdir="stdmet", suffix="h" for bulk parameters).

    Parameters
    ----------
    buoy_id : str
        NDBC buoy identifier.
    subdir : str
        Historical data subdirectory (e.g. "stdmet", "swden", "swdir").
    suffix : str
        File suffix used in NDBC's historical filenames (e.g. "h", "w", "d").

    Returns
    -------
    List[int]
        Sorted list of years with available data.
    """

    url = f"https://www.ndbc.noaa.gov/station_history.php?station={buoy_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    pattern = rf"{buoy_id}{suffix}(\d{{4}})\.txt\.gz&dir=data/historical/{subdir}/"
    return sorted(set(int(year) for year in re.findall(pattern, response.text)))


def get_available_directional_years(buoy_id: str) -> List[int]:
    """
    Find which years have all four directional-spectrum coefficient files
    (alpha1, alpha2, r1, r2) available for a buoy.

    Parameters
    ----------
    buoy_id : str
        NDBC buoy identifier.

    Returns
    -------
    List[int]
        Sorted list of years where all four coefficients are available.
    """

    d_years = get_available_years(buoy_id, subdir="swdir", suffix="d")
    i_years = get_available_years(buoy_id, subdir="swdir2", suffix="i")
    j_years = get_available_years(buoy_id, subdir="swr1", suffix="j")
    k_years = get_available_years(buoy_id, subdir="swr2", suffix="k")

    return sorted(set(d_years) & set(i_years) & set(j_years) & set(k_years))


def select_years(
    available_years: List[int],
    download_mode: str,
    buoy_id: str,
    label: str,
    single_year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> List[int]:
    """
    Apply the notebook-wide download_mode/single_year/start_year/end_year
    selection to a list of available years, so every dataset (bulk
    parameters, energy spectra, directional spectra) is restricted to the
    same requested period. Raises if the selection is empty, and prints a
    one-line summary of the years selected otherwise.

    Parameters
    ----------
    available_years : List[int]
        Years for which data is available (from `get_available_years` or
        `get_available_directional_years`).
    download_mode : str
        One of "all", "period", or "single".
    buoy_id : str
        NDBC buoy identifier, used in the printed/raised messages.
    label : str
        Short description of the dataset being selected (e.g. "year(s)
        selected", "years of spectra selected"), used in the printed message.
    single_year : int, optional
        Required when download_mode="single".
    start_year : int, optional
        Required when download_mode="period".
    end_year : int, optional
        Required when download_mode="period".

    Returns
    -------
    List[int]
        The selected years.
    """

    if download_mode == "all":
        years = available_years
    elif download_mode == "period":
        years = [y for y in available_years if start_year <= y <= end_year]
    elif download_mode == "single":
        years = [y for y in available_years if y == single_year]
    else:
        raise ValueError(f"Unknown download_mode: {download_mode!r}")

    if not years:
        raise ValueError(
            f"No available years for buoy {buoy_id} match download_mode={download_mode!r} "
            f"(available: {available_years[0]}-{available_years[-1]})"
        )

    print(f"Buoy {buoy_id}: {len(years)} {label} -> {years}")
    return years


def read_wave_spectra_full(base_path: str, buoy_id: str, years: List[int]) -> pd.DataFrame:
    """
    Read all yearly wave spectra CSVs into one DataFrame indexed by date.

    NDBC's date columns changed over the years ("YY"/"YYYY"/"#YY" for the year,
    with or without a minute column), so each file is inspected individually
    instead of assuming a single fixed layout.

    Parameters
    ----------
    base_path : str
        Root download path (`NOAADownloader.base_path_to_download`).
    buoy_id : str
        NDBC buoy identifier.
    years : List[int]
        Years to read.

    Returns
    -------
    pd.DataFrame
        Wave energy spectra, indexed by datetime, columns = frequency bins (Hz).
    """

    frames = []
    for year in years:
        file_path = os.path.join(
            base_path,
            "NDBC",
            "buoy_data",
            buoy_id,
            "wave_spectra",
            f"buoy_{buoy_id}_spectra_{year}.csv",
        )
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"No wave spectra file found for buoy {buoy_id} year {year}")
            continue

        if "YYYY" in df.columns:
            year_col = "YYYY"
        elif "#YY" in df.columns:
            year_col = "#YY"
        else:
            year_col = "YY"
            df[year_col] = df[year_col] + 1900  # two-digit legacy year

        id_cols = [year_col, "MM", "DD", "hh"] + (["mm"] if "mm" in df.columns else [])
        rename = {year_col: "year", "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"}

        df["date"] = pd.to_datetime(df[id_cols].rename(columns=rename))
        frames.append(df.drop(columns=id_cols).set_index("date"))

    return pd.concat(frames).sort_index()


def get_station_coords(buoy_id: str) -> Tuple[float, float]:
    """
    Parse a buoy's (latitude, longitude) in degrees from its NDBC station page.

    Parameters
    ----------
    buoy_id : str
        NDBC buoy identifier.

    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in degrees, with N/E positive and S/W negative.
    """

    url = f"https://www.ndbc.noaa.gov/station_page.php?station={buoy_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    match = re.search(r"\((\d+\.\d+)(N|S) (\d+\.\d+)(E|W)\)", response.text)
    if not match:
        raise ValueError(f"Could not find coordinates for buoy {buoy_id}")

    lat = float(match.group(1)) * (1 if match.group(2) == "N" else -1)
    lon = float(match.group(3)) * (1 if match.group(4) == "E" else -1)
    return lat, lon


def align_directional_coefficients(*frames: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Match all five coefficient tables by timestamp and numeric frequency.

    Keep only shared labels, in sorted order. Missing coefficient values mask
    that time-frequency bin in every table; no missing directions are filled.
    """
    if len(frames) != 5:
        raise ValueError("Expected alpha1, alpha2, r1, r2 and c11 tables.")
    prepared = []
    for name, frame in zip(("alpha1", "alpha2", "r1", "r2", "c11"), frames):
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(frame.index)
        frame.columns = pd.Index([float(value) for value in frame.columns])
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"{name}: duplicate timestamps or frequency bins.")
        frame = frame.astype(float)
        prepared.append(frame.where(np.isfinite(frame) & (frame < 999)))
    times, frequencies = prepared[0].index, prepared[0].columns
    for frame in prepared[1:]:
        times = times.intersection(frame.index)
        frequencies = frequencies.intersection(frame.columns)
    times, frequencies = times.sort_values(), frequencies.sort_values()
    if times.empty or frequencies.empty:
        raise ValueError("Directional coefficients have no shared timestamps or frequencies.")
    aligned = [frame.loc[times, frequencies] for frame in prepared]
    valid = np.logical_and.reduce([frame.notna().to_numpy() for frame in aligned])
    keep_times = valid.any(axis=1)
    if not keep_times.any():
        raise ValueError("No complete directional coefficient observations on the shared grid.")
    return tuple(frame.where(valid).loc[keep_times] for frame in aligned)


def build_directional_spectrum_dataset(
    alpha1_df: pd.DataFrame,
    alpha2_df: pd.DataFrame,
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    c11_df: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> xr.Dataset:
    """
    Reconstruct the full directional wave spectrum from NDBC's Fourier
    coefficient files, as an "offshore_spectra"-format Dataset (dims: time,
    freq, dir; variable: efth; scalar coords: latitude, longitude).

    Parameters
    ----------
    alpha1_df, alpha2_df, r1_df, r2_df, c11_df : pd.DataFrame
        Directional spectrum coefficients, each with a datetime index and
        frequency (Hz) columns, as returned by
        `bluemath_tk.downloaders.noaa.noaa_downloader.read_directional_spectra`.
    latitude : float
        Buoy latitude in degrees.
    longitude : float
        Buoy longitude in degrees (any convention; converted to 0-360).

    Returns
    -------
    xr.Dataset
        Directional wave spectrum with data variable "efth" (m^2/Hz/deg).
    """

    alpha1_df, alpha2_df, r1_df, r2_df, c11_df = align_directional_coefficients(
        alpha1_df, alpha2_df, r1_df, r2_df, c11_df,
    )

    # NDBC's fill value for missing observations; mask it out before reconstructing.
    alpha1_vals = alpha1_df.values.copy()
    alpha2_vals = alpha2_df.values.copy()
    r1_vals = r1_df.values.copy()
    r2_vals = r2_df.values.copy()
    c11_vals = c11_df.values.copy()
    for arr in (alpha1_vals, alpha2_vals, r1_vals, r2_vals, c11_vals):
        arr[arr >= 999] = np.nan

    freq = np.array([float(col) for col in c11_df.columns])
    dir_deg = np.arange(0, 360, 5)  # deg, direction waves come from

    r1 = r1_vals / 100.0  # (time, freq)
    r2 = r2_vals / 100.0
    alpha1_rad = np.deg2rad(alpha1_vals)
    alpha2_rad = np.deg2rad(alpha2_vals)
    theta_rad = np.deg2rad(dir_deg)  # (dir,)

    cos1 = np.cos(theta_rad[None, None, :] - alpha1_rad[:, :, None])
    cos2 = np.cos(2 * (theta_rad[None, None, :] - alpha2_rad[:, :, None]))
    D = (1 / np.pi) * (0.5 + r1[:, :, None] * cos1 + r2[:, :, None] * cos2)

    # Normalize D to integrate to 1 over the (periodic, evenly-spaced) direction axis
    dtheta = 2 * np.pi / len(dir_deg)
    D = D / (np.nansum(D, axis=-1, keepdims=True) * dtheta)
    D[D < 0] = 0

    efth = c11_vals[:, :, None] * D  # (time, freq, dir), m^2/Hz/rad

    # The "dir" coordinate is stored in degrees (readable, CF-friendly), so efth
    # must be converted from per-radian to per-degree density -- otherwise any
    # integration over "dir" using the coordinate's own values (e.g. xarray's
    # `.integrate("dir")`, used to recover Hs = 4*sqrt(m0)) silently mixes units
    # and comes out ~57x (180/pi) too large.
    efth = efth * (np.pi / 180.0)  # m^2/Hz/rad -> m^2/Hz/deg

    ds_out = xr.Dataset(
        data_vars={"efth": (("time", "freq", "dir"), efth)},
        coords={
            "time": c11_df.index.values,
            "freq": freq,
            "dir": dir_deg,
            "latitude": latitude,
            "longitude": longitude % 360,
        },
    )
    ds_out["freq"].attrs = {"units": "Hz", "standard_name": "sea_surface_wave_frequency"}
    ds_out["dir"].attrs = {
        "units": "degree",
        "standard_name": "sea_surface_wave_from_direction",
    }
    ds_out["efth"].attrs = {
        "units": "m2 Hz-1 deg-1",
        "standard_name": "sea_surface_wave_directional_variance_spectral_density",
    }

    return ds_out


def save_netcdf(ds: xr.Dataset, output_path: str, encoding: Optional[dict] = None) -> None:
    """
    Save a Dataset to NetCDF, working around filesystems that
    don't support the flock() locking HDF5 uses by default -- which would
    otherwise turn a direct `ds.to_netcdf(output_path)` into a spurious
    PermissionError. Writes to local scratch space first and moves the
    finished file into place, so an HDF5 file is never opened for writing
    directly on such a mount.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to save.
    output_path : str
        Final destination path for the NetCDF file.
    encoding : dict, optional
        Passed through to `xr.Dataset.to_netcdf`.
    """

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    ds.to_netcdf(tmp_path, encoding=encoding)

    shutil.move(tmp_path, output_path)
    os.chmod(output_path, 0o644)  # NamedTemporaryFile defaults to 0600
