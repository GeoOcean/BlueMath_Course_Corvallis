from typing import List, Optional, Union

import matplotlib.pyplot as plt
import xarray as xr


def plot_bathymetry(
    elevation: xr.DataArray,
    depth_contours: Optional[Union[float, List[float]]] = 150,
    contour_color: str = "red",
    title: str = "GEBCO 2025",
    figsize: tuple = (6, 6),
) -> plt.Axes:
    """
    Plot GEBCO bathymetry with coastline and depth contours.

    Parameters
    ----------
    elevation : xr.DataArray
        Elevation data with 'lon' and 'lat' coordinates
    depth_contours : float or list of float, optional
        Depth(s) in meters (positive values) to draw as contour lines.
        Default is 150.
    contour_color : str, optional
        Color of the depth contour lines. Default is "red".
    title : str, optional
        Plot title. Default is "GEBCO 2025".
    figsize : tuple, optional
        Figure size. Default is (6, 6).

    Returns
    -------
    plt.Axes
        Axes containing the plot
    """

    if depth_contours is None:
        depth_contours = []
    elif isinstance(depth_contours, (int, float)):
        depth_contours = [depth_contours]

    fig, ax = plt.subplots(figsize=figsize)
    elevation.plot(ax=ax, cmap="terrain", center=0)

    # Coastline: contour at elevation = 0 (sea level)
    ax.contour(
        elevation["lon"], elevation["lat"], elevation,
        levels=[0], colors="black", linewidths=1
    )

    # Depth contours (water)
    if depth_contours:
        levels = sorted(-abs(d) for d in depth_contours)
        cs = ax.contour(
            elevation["lon"], elevation["lat"], elevation,
            levels=levels, colors=contour_color, linewidths=1, linestyles="--"
        )
        ax.clabel(cs, fmt="%d m", fontsize=8)

    ax.set_title(title)

    return ax
