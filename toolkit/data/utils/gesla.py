from io import BytesIO
from pathlib import Path
import re
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr

ERDDAP_URL = "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_hourly_gesla"

STATION_COLUMNS = [
    "station_name",
    "agency_id",
    "station_code",
    "station_country_code",
    "latitude",
    "longitude",
    "record_id",
]
DATA_COLUMNS = ["sea_level", "time", "flag1", "flag2", "record_id"]


def find_nearby_stations(
    latitude: float,
    longitude: float,
    limit: int = 10,
    erddap_url: str = ERDDAP_URL,
) -> pd.DataFrame:
    """Return the nearest stations to a location, ordered by distance in km."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    query = f"{','.join(STATION_COLUMNS)}&distinct()"
    response = requests.get(f"{erddap_url}.csv0?{query}", timeout=120)
    response.raise_for_status()
    if not response.content.strip():
        raise RuntimeError("ERDDAP returned an empty station catalog.")

    stations = pd.read_csv(
        BytesIO(response.content),
        header=None,
        names=STATION_COLUMNS,
    )
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude"]).copy()
    if stations.empty:
        raise RuntimeError("The station catalog contains no valid coordinates.")

    earth_radius_km = 6371.0088
    reference_latitude = np.radians(latitude)
    station_latitudes = np.radians(stations["latitude"].to_numpy())
    latitude_difference = station_latitudes - reference_latitude
    longitude_difference_degrees = (
        (stations["longitude"].to_numpy() - longitude + 180) % 360
    ) - 180
    longitude_difference = np.radians(longitude_difference_degrees)
    haversine_value = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(reference_latitude)
        * np.cos(station_latitudes)
        * np.sin(longitude_difference / 2) ** 2
    )
    stations["distance_km"] = (
        2 * earth_radius_km * np.arcsin(np.sqrt(np.clip(haversine_value, 0, 1)))
    )

    return stations.sort_values("distance_km").head(limit).reset_index(drop=True)


def download_sea_level_data(
    record_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    erddap_url: str = ERDDAP_URL,
) -> pd.DataFrame:
    """Download one GESLA record, retaining both source quality flags."""
    query = f"{','.join(DATA_COLUMNS)}&" + quote(
        f'record_id="{record_id}"', safe=""
    )
    if start_date is not None:
        query += "&" + quote(f"time>={start_date}", safe=":-TZ")
    if end_date is not None:
        query += "&" + quote(f"time<={end_date}", safe=":-TZ")

    response = requests.get(f"{erddap_url}.csv0?{query}", timeout=300)
    response.raise_for_status()
    if not response.content.strip():
        raise RuntimeError("ERDDAP returned no observations for this station.")

    observations = pd.read_csv(
        BytesIO(response.content),
        header=None,
        names=DATA_COLUMNS,
    )
    observations["time"] = pd.to_datetime(
        observations["time"], utc=True, errors="coerce"
    )
    observations["sea_level"] = pd.to_numeric(
        observations["sea_level"], errors="coerce"
    ).replace(-99.9999, np.nan)
    for flag_name in ("flag1", "flag2"):
        observations[flag_name] = pd.to_numeric(
            observations[flag_name], errors="coerce"
        ).astype("Int64")

    return observations


def plot_sea_level(
    observations: pd.DataFrame,
    station_name: str,
) -> plt.Axes | None:
    """Plot observations marked for use by the GESLA quality flag."""
    usable = observations.loc[
        observations["sea_level"].notna() & observations["flag2"].eq(1)
    ]
    if usable.empty:
        print("No observations marked for use are available to plot.")
        return None

    axes = usable.plot(
        x="time",
        y="sea_level",
        figsize=(13, 4),
        linewidth=0.7,
        legend=False,
    )
    axes.set(
        title=f"Sea Level at {station_name}",
        xlabel="Time (UTC)",
        ylabel="Sea level (m)",
    )
    axes.grid(True, alpha=0.25)
    axes.figure.tight_layout()
    plt.show()
    return axes


def save_sea_level_data(
    observations: pd.DataFrame,
    station: pd.Series,
    output_directory: str | Path = "outputs",
) -> Path:
    """Save all observations and source flags to a station-named NetCDF file."""
    station_name = str(station["station_name"])
    station_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", station_name).strip("_")
    filename = (
        f"GESLA_{station_slug}_{station['station_code']}_{station['agency_id']}.nc"
    )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / filename

    time = observations["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    dataset = xr.Dataset(
        data_vars={
            "sea_level": ("time", observations["sea_level"].to_numpy(dtype=float)),
            "flag1": ("time", observations["flag1"].fillna(-1).to_numpy(dtype="int8")),
            "flag2": ("time", observations["flag2"].fillna(-1).to_numpy(dtype="int8")),
        },
        coords={"time": time.to_numpy()},
    )
    dataset["sea_level"].attrs = {
        "long_name": "Sea level",
        "units": "m",
        "standard_name": "sea_surface_height",
    }
    dataset["flag1"].attrs = {
        "long_name": "GESLA quality-control flag",
        "_FillValue": np.int8(-1),
    }
    dataset["flag2"].attrs = {
        "long_name": "GESLA use flag (1 = use, 0 = do not use)",
        "_FillValue": np.int8(-1),
    }
    dataset.attrs = {
        "title": f"GESLA sea-level record at {station_name}",
        "source": ERDDAP_URL,
        "record_id": str(station["record_id"]),
        "station_name": station_name,
        "station_code": str(station["station_code"]),
        "agency_id": str(station["agency_id"]),
        "latitude": float(station["latitude"]),
        "longitude": float(station["longitude"]),
    }
    dataset.to_netcdf(output_path, encoding={"time": {"units": "hours since 1900-01-01"}})
    return output_path
