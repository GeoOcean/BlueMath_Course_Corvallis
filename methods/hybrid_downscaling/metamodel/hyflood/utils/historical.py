import numpy as np
import pandas as pd
import xarray as xr

from .reconstruction import pcs_to_flood_maps


def load_gesla_sea_level(path, datum_offset=0.0):
    """
    Load an hourly GESLA sea-level record (as saved by `GESLA_data.ipynb`).

    Parameters
    ----------
    path : str
        Path to the GESLA NetCDF file.
    datum_offset : float, optional
        Value [m] subtracted from the record to change its vertical datum
        (e.g. station datum -> NAVD88).

    Returns
    -------
    pd.Series
        Hourly sea level [m], keeping only the observations that GESLA marks
        for use (`flag2 == 1`).
    """
    ds = xr.open_dataset(path)

    sea_level = ds["sea_level"].where(ds["flag2"] == 1).to_series().dropna()

    return (sea_level - datum_offset).rename("SL")


def offshore_wave_parameters(spectra, read_size=10000, chunk_size=1000):
    """
    Bulk wave parameters of an offshore directional spectrum.

    Parameters
    ----------
    spectra : xr.Dataset
        Directional spectra with variable `efth` [m2/Hz/deg] and dimensions
        (time, freq, dir).
    read_size : int, optional
        Number of time steps read from disk at once. Large blocks avoid
        decompressing the same NetCDF chunk many times.
    chunk_size : int, optional
        Number of time steps processed at once, to limit memory usage.

    Returns
    -------
    pd.DataFrame
        Significant wave height `Hs` [m], peak period `Tp` [s] and peak
        direction `Dir` [deg] (coming from, nautical), one row per time step.
    """
    efth = spectra["efth"].transpose("time", "freq", "dir")
    freq = spectra["freq"].values
    theta = np.deg2rad(spectra["dir"].values)
    dfreq = np.gradient(freq)
    ddir = 360 / spectra.sizes["dir"]

    hs, tp, dpm = [], [], []

    for read_start in range(0, spectra.sizes["time"], read_size):
        block = efth.isel(time=slice(read_start, read_start + read_size)).values

        for start in range(0, len(block), chunk_size):
            e = block[start : start + chunk_size]

            # Frequency spectrum (integrated over directions) and its peak
            ef = e.sum(axis=2) * ddir
            ipeak = np.argmax(np.nan_to_num(ef, nan=-1), axis=1)

            # Hs = 4 sqrt(m0)
            hs.append(4 * np.sqrt((ef * dfreq).sum(axis=1)))
            tp.append(1 / freq[ipeak])

            # Mean direction of the energy at the peak frequency
            e_peak = e[np.arange(len(e)), ipeak, :]
            dpm.append(
                np.rad2deg(
                    np.arctan2((e_peak * np.sin(theta)).sum(axis=1), (e_peak * np.cos(theta)).sum(axis=1))
                ) % 360
            )

    waves = pd.DataFrame(
        {"Hs": np.concatenate(hs), "Tp": np.concatenate(tp), "Dir": np.concatenate(dpm)},
        index=spectra.get_index("time"),
    ).dropna()

    # Remove empty or corrupted spectra (no energy, or peak at the lowest frequency bin)
    valid = (waves["Hs"] > 0.1) & (waves["Tp"] < 1 / freq[0])

    return waves[valid]


def build_daily_events(sea_level, waves, precipitation, rain_threshold=0.1):
    """
    Build one compound flooding event per day.

    Each event is defined at the daily high tide:

    - `SL`: daily maximum sea level.
    - `Hs`, `Tp`, `Dir`: offshore waves at the time of the daily maximum
      sea level (nearest record within 1 hour).
    - `precipitation`: total rainfall of the day [mm].
    - `precip_duration`: number of hours of the day with rain above
      `rain_threshold` [mm/h] (at least 1 h).

    Parameters
    ----------
    sea_level : pd.Series
        Hourly sea level [m].
    waves : pd.DataFrame
        Offshore wave parameters (`Hs`, `Tp`, `Dir`).
    precipitation : pd.Series
        Hourly precipitation [mm].
    rain_threshold : float, optional
        Minimum hourly rainfall [mm] counted as a rainy hour.

    Returns
    -------
    pd.DataFrame
        One row per day in which all the drivers are available.
    """
    daily_sl = sea_level.groupby(sea_level.index.floor("D"))
    high_tide_time = daily_sl.idxmax()

    waves_at_high_tide = waves.sort_index().reindex(
        high_tide_time.values, method="nearest", tolerance=pd.Timedelta("1h")
    )
    waves_at_high_tide.index = high_tide_time.index

    rain_daily = precipitation.resample("1D")

    events = pd.DataFrame(
        {
            "Hs": waves_at_high_tide["Hs"],
            "Tp": waves_at_high_tide["Tp"],
            "Dir": waves_at_high_tide["Dir"],
            "SL": daily_sl.max(),
            "precipitation": rain_daily.sum(),
            "precip_duration": (precipitation > rain_threshold)
            .resample("1D")
            .sum()
            .clip(lower=1),
        }
    )
    events.index.name = "time"

    return events.dropna()


def clip_to_training_range(events, training, directional_variables=("Dir",)):
    """
    Clip the events to the range of the training scenarios.

    Directional variables are moved to the closest bound along the circle
    (e.g. 10° goes to 360° and not to 160°).

    Returns
    -------
    pd.DataFrame
        Clipped events.
    pd.Series
        Fraction of events outside the training range, per variable.
    """
    lower, upper = training.min(), training.max()
    clipped = events.copy()

    # Ignore tiny departures (e.g. a dry day vs. the 0.001 mm minimum of the LHS)
    tolerance = 0.01 * (upper - lower)
    outside = pd.Series(
        {
            var: (
                (events[var] < lower[var] - tolerance[var])
                | (events[var] > upper[var] + tolerance[var])
            ).mean()
            for var in training.columns
        }
    )

    for var in training.columns:
        if var in directional_variables:
            values = events[var] % 360
            out = (values < lower[var]) | (values > upper[var])
            dist_lower = (lower[var] - values) % 360
            dist_upper = (values - upper[var]) % 360
            nearest = np.where(dist_lower < dist_upper, lower[var], upper[var])
            clipped[var] = np.where(out, nearest, values)
        else:
            clipped[var] = events[var].clip(lower[var], upper[var])

    return clipped, outside


def historical_flood_statistics(pca, pcs, threshold=0.1, chunk_size=50):
    """
    Reconstruct the daily flood maps and accumulate their statistics.

    The maps are reconstructed in chunks, so that the full set of daily maps
    is never held in memory at the same time.

    Parameters
    ----------
    pca : PCA
        Fitted bluemath_tk PCA model.
    pcs : pd.DataFrame
        Predicted PCs, indexed by day.
    threshold : float, optional
        Minimum flood depth [m] for a cell to be considered flooded.
    chunk_size : int, optional
        Number of days reconstructed at once.

    Returns
    -------
    xr.Dataset
        - `wet_days` (year, y, x): number of flooded days per year.
        - `annual_max_depth` (year, y, x): maximum flood depth per year [m].
        - `n_days` (year): number of reconstructed days per year.
        - `flooded_area` (time): daily flooded area [km2].
    """
    days = pd.DatetimeIndex(pcs.index)
    years = np.unique(days.year)

    wet_days, annual_max, areas = None, None, []

    for start in range(0, len(pcs), chunk_size):
        maps = pcs_to_flood_maps(pca, pcs.iloc[start : start + chunk_size].reset_index(drop=True))
        chunk_years = days.year[start : start + chunk_size]

        if wet_days is None:
            template = maps.isel(case_num=0, drop=True)
            land = template.notnull()  # NaN outside the domain / below MHHW
            pixel_area = abs(float(maps.x[1] - maps.x[0]) * float(maps.y[1] - maps.y[0]))
            wet_days = np.zeros((len(years),) + template.shape, dtype="int32")
            annual_max = np.zeros((len(years),) + template.shape, dtype="float32")

        depth = np.nan_to_num(maps.values.astype("float32"))
        flooded = depth > threshold
        areas.append(flooded.sum(axis=(1, 2)) * pixel_area / 1e6)

        for year in np.unique(chunk_years):
            iy = np.searchsorted(years, year)
            in_year = chunk_years == year
            wet_days[iy] += flooded[in_year].sum(axis=0)
            annual_max[iy] = np.maximum(annual_max[iy], depth[in_year].max(axis=0))

    dims = ("year",) + template.dims
    coords = {"year": years, **{dim: template[dim] for dim in template.dims}}

    return xr.Dataset(
        {
            "wet_days": xr.DataArray(wet_days, dims=dims, coords=coords).where(land),
            "annual_max_depth": xr.DataArray(annual_max, dims=dims, coords=coords).where(land),
            "n_days": ("year", pd.Series(days.year).value_counts().reindex(years).values),
            "flooded_area": ("time", np.concatenate(areas)),
        },
        coords={"time": days},
    )
