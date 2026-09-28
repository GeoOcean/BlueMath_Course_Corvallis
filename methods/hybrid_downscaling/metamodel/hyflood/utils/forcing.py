
import numpy as np
import pandas as pd
import xarray as xr

def _resample_timeseries(series, dt_seconds):
    """Linearly interpolate a time series to a regular time step."""
    time = pd.date_range(
        series.index[0],
        series.index[-1],
        freq=f"{dt_seconds}s",
    )

    return (
        series.reindex(series.index.union(time))
        .interpolate(method="time")
        .reindex(time)
    )


def _add_zero_warmup(series, warmup_hours, dt_seconds):
    """Prepend a zero-valued warm-up period."""
    if warmup_hours <= 0:
        return series

    n_steps = round(warmup_hours * 3600 / dt_seconds)
    offset = pd.Timedelta(seconds=n_steps * dt_seconds)

    series = series.copy()
    series.index += offset

    warmup_index = pd.date_range(
        start=series.index[0] - offset,
        periods=n_steps,
        freq=f"{dt_seconds}s",
    )

    return pd.concat([
        pd.Series(0.0, index=warmup_index),
        series,
    ])


def _build_ig_signal(h_ig, period_seconds, dt_seconds, phase=0.0):
    """Convert infragravity wave height into a harmonic water-level signal."""
    amplitude = _resample_timeseries(
        h_ig / (2 * np.sqrt(2)),
        dt_seconds,
    )

    time = (amplitude.index - amplitude.index[0]).total_seconds()

    return amplitude * np.sin(
        2 * np.pi * time / period_seconds + phase
    )

def build_waterlevel_forcings(
    msetup,
    h_ig,
    msl,
    time,
    reference_time,
    ig_period=200,
    dt_seconds=60,
    warmup_hours=0,
    phase=0.0,
):
    """
    Build SFINCS water-level forcings for a single boundary point.

    Parameters
    ----------
    msetup : array-like
        Mean wave setup with shape (n_cases, n_times).
    h_ig : array-like
        Infragravity wave height with shape (n_cases, n_times).
    msl : array-like
        Mean sea level for each case.
    time : array-like
        Time in hours relative to `reference_time`.
    reference_time : str or pd.Timestamp
        Reference time for the SFINCS simulation.

    Returns
    -------
    slow_forcings : list[pd.DataFrame]
        Slowly varying water-level forcing (MSL + wave setup).
    fast_forcings : list[pd.DataFrame]
        Infragravity water-level forcing.
    """
    index = (
        pd.Timestamp(reference_time)
        + pd.to_timedelta(time, unit="h")
    )

    slow_forcings = []
    fast_forcings = []

    for setup, ig, sea_level in zip(msetup, h_ig, msl):
        setup = pd.Series(setup, index=index)
        ig = pd.Series(ig, index=index)

        slow = _resample_timeseries(setup, dt_seconds) + sea_level
        fast = _build_ig_signal(
            ig,
            period_seconds=ig_period,
            dt_seconds=dt_seconds,
            phase=phase,
        )

        slow_forcings.append(
            _add_zero_warmup(slow, warmup_hours, dt_seconds).to_frame()
        )
        fast_forcings.append(
            _add_zero_warmup(fast, warmup_hours, dt_seconds).to_frame()
        )

    return slow_forcings, fast_forcings

def build_precipitation_forcings(
    centroids,
    reference_time,
    dt_minutes=60,
):
    """Build precipitation forcings for all MDA cases."""
    return [
        build_precipitation_forcing(
            precipitation=row.precipitation,
            duration=row.precip_duration,
            lag=row.precip_wave_lag,
            reference_time=reference_time,
            dt_minutes=dt_minutes,
        )
        for _, row in centroids.iterrows()
    ]

def build_precipitation_forcing(
    precipitation,
    duration,
    lag,
    reference_time,
    wave_peak_hour=12,
    dt_minutes=60,
):
    """
    Build a triangular precipitation hyetograph for SFINCS.

    Parameters
    ----------
    precipitation : float
        Total precipitation depth in mm.
    duration : float
        Storm duration in hours.
    lag : float
        Time lag in hours between the precipitation and wave peaks.
        Negative values indicate that precipitation peaks before the waves.
    reference_time : str or pd.Timestamp
        Reference date of the simulation.
    wave_peak_hour : float, optional
        Hour of the wave peak relative to the reference time.
    dt_minutes : int, optional
        Temporal resolution of the forcing in minutes.

    Returns
    -------
    pd.DataFrame
        Precipitation timeseries in mm/h.
    """
    reference_time = pd.Timestamp(reference_time)

    peak_time = reference_time + pd.Timedelta(
        hours=wave_peak_hour + lag
    )

    start_time = peak_time - pd.Timedelta(hours=duration / 2)
    end_time = peak_time + pd.Timedelta(hours=duration / 2)

    time = pd.date_range(
        start_time,
        end_time,
        freq=f"{dt_minutes}min",
    )

    hours_from_peak = (
        time - peak_time
    ).total_seconds() / 3600

    shape = np.maximum(
        1 - 2 * np.abs(hours_from_peak) / duration,
        0,
    )

    # Scale so that the integral equals total precipitation
    intensity = shape * precipitation / np.trapezoid(
        shape,
        x=(time - time[0]).total_seconds() / 3600,
    )

    df = pd.DataFrame(
        {"precip": intensity},
        index=time,
    )

    df.index.name = "time"

    return df