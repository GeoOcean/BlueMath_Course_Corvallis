import ipywidgets as widgets
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.plotting.scatter import plot_scatters_in_triangle
from bluemath_tk.datamining.pca import PCA
from bluemath_tk.interpolation.rbf import RBF
from bluemath_tk.waves.series import waves_dispersion
from ipywidgets import interact

def animate_case_propagation(
    case_dataset, depth, tini=0, tend=30, tstep=2, figsize=(15, 5), dxinp=1
):
    """Animate water elevation above a fixed bed.

    Parameters
    ----------
    case_dataset : xarray.Dataset
        Watlev in meters, with a Tsec dimension and one value per bed node.
        Tsec coordinates provide the displayed time in seconds.
    depth : array-like
        Bed depths in meters, positive downward from the water reference level.
    tini, tend : int, optional
        First frame index (inclusive) and last frame index (exclusive).
    tstep : int, optional
        Positive frame index increment.
    figsize : tuple, optional
        Figure width and height in inches.
    dxinp : float, optional
        Horizontal node spacing in meters.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation suitable for display in Jupyter or saving to a file.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_depthfile(depth, ax=ax, dxinp=dxinp)
    ax.collections[-1].remove()  # Replace still water with the animated surface.
    bed = -np.asarray(depth, dtype=float)
    x = np.arange(bed.size) * dxinp
    # Reuse one polygon: only its upper edge changes between frames.
    vertices = np.column_stack((np.r_[x, x[::-1]], np.r_[bed, bed[::-1]]))
    water, = ax.fill(vertices[:, 0], vertices[:, 1],
                     color="deepskyblue", edgecolor='wheat', linewidth=1, alpha=0.5, zorder=3)
    title = ax.set_title("")
    ax.set_ylim(min(bed.min(), -10), max(bed.max(), 6))

    def update(frame):
        elevation = np.asarray(case_dataset["Watlev"].isel(Tsec=frame)).reshape(-1)
        if elevation.size != bed.size:
            raise ValueError("Watlev must contain one value per bed node.")
        # SWASH can return NaN for nodes that are dry; those nodes have no
        # water surface and must remain on the bed in the filled polygon.
        surface = np.where(np.isfinite(elevation), np.maximum(elevation, bed), bed)
        vertices[:bed.size, 1] = surface
        water.set_xy(vertices)
        title.set_text(f"Time: {case_dataset['Tsec'].values[frame]:g} s")
        return water, title

    ani = animation.FuncAnimation(
        fig, update, frames=range(tini, tend, tstep),
        init_func=lambda: (water, title), blit=True,
    )
    plt.close(fig)
    return ani


def show_graph_for_different_parameters(pca: PCA, rbf: RBF, lhs_parameters, depthfile):
    """
    Display an interactive wave reconstruction above the reef profile.

    Parameters
    ----------
    pca : PCA
        Fitted PCA model used to reconstruct spatial wave heights.
    rbf : RBF
        Fitted interpolator predicting PCA scores from the slider values.
    lhs_parameters : dict
        Matching dimensions_names, lower_bounds, and upper_bounds sequences
        for Hs (m), Hs_L0 (dimensionless), Wv (m), hv (m), and Nv (stems/m2).
    depthfile : str or pathlib.Path
        Depth file in meters, positive downward, with 1 m node spacing.
    """
    depth = np.loadtxt(depthfile)

    # Function to update the plot based on widget input
    def update_plot(Hs, Hs_L0, Wv, hv, Nv):
        data = {"Hs": Hs, "Hs_L0": Hs_L0, "Wv": Wv, "hv": hv, "Nv": Nv}
        df_dataset_single_case = pd.DataFrame([data])

        # Spatial Reconstruction
        predicted_hs = rbf.predict(dataset=df_dataset_single_case)
        predicted_hs_ds = xr.Dataset(
            {
                "PCs": (["case_num", "n_component"], predicted_hs.values),
            },
            coords={
                "case_num": [0],
                "n_component": np.arange(len(pca.pcs_df.columns)),
            },
        )
        ds_output_all = pca.inverse_transform(PCs=predicted_hs_ds)

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(11, 3))
        plot_depthfile(depth, ax=ax)
        ds_output_all["Hs"].sel(case_num=0).plot(x="Xp", ax=ax, color="k")
        ax.set_title("")

        # Improved vegetation visualization using fill_between
        veg_start = 400 - int(Wv)
        veg_end = 400

        n_stems = int(Nv / 10)  # Number of stems based on density
        stem_positions = np.linspace(veg_start, veg_end - 1, n_stems).astype(int)
        y_base = -depth[stem_positions]
        ax.vlines(stem_positions, y_base, y_base + hv,
                  color="green", linewidth=2, alpha=0.8, zorder=4)

        ax.set_ylim(-12, 6)
        ax.set_xlim(0, 600)
        ax.grid(True)

    parameters = {}
    for name, lower, upper in zip(
        lhs_parameters["dimensions_names"],
        lhs_parameters["lower_bounds"],
        lhs_parameters["upper_bounds"],
    ):
        parameters[name] = widgets.FloatSlider(
            value=np.random.uniform(lower, upper),
            min=lower, max=upper, step=(upper - lower) / 10,
            description=name, continuous_update=False,
        )
    return interact(update_plot, **{
        name: parameters[name] for name in ("Hs", "Hs_L0", "Wv", "hv", "Nv")
    })


def plot_depthfile(depth=None, depthfile=None, ax=None, xlim=None, dxinp=1):
    """Plot bed elevation and still water from a depth array or text file.

    Parameters
    ----------
    depth : array-like, optional
        One-dimensional depths in meters, positive downward; negative values
        represent exposed ground. Required when depthfile is omitted.
    depthfile : str or pathlib.Path, optional
        Text file containing depths in meters. Overrides depth when provided.
    ax : matplotlib.axes.Axes, optional
        Axes to reuse; otherwise create an 11 by 3 inch figure.
    xlim : tuple, optional
        Horizontal limits in meters. Defaults to the full profile.
    dxinp : float, optional
        Positive horizontal node spacing in meters.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the profile, ready for additional curves or styling.
    """
    depth = np.asarray(
        np.loadtxt(depthfile) if depthfile is not None else depth, dtype=float
    )
    if depth.ndim != 1 or depth.size < 2 or not np.isfinite(depth).all():
        raise ValueError("Provide at least two finite depths in a 1D array or file.")
    if not np.isfinite(dxinp) or dxinp <= 0:
        raise ValueError("dxinp must be positive and finite.")

    x = np.arange(depth.size) * dxinp
    bed = -depth
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(x, bed.min(), bed, color="wheat", zorder=2)
    ax.fill_between(x, bed, 0, where=depth >= 0, interpolate=True,
                    color="deepskyblue", alpha=0.5, zorder=1)
    ax.set(xlim=(x[0], x[-1]) if xlim is None else xlim,
           ylim=(bed.min(), None), xlabel="Distance from offshore (m)", ylabel="Elevation (m)")
    #for spine in ax.spines.values():
    #    spine.set_visible(False)
    return ax


def plot_scatters_Tp(df_centroids, df_lhs_data, scatter_points_thick=10):
    """
    Plot peak wave period (Tp) scatter plots in triangular format.

    This function calculates the peak wave period (Tp) from significant wave height (Hs)
    and wave steepness (Hs_L0) using the deep water wave dispersion relation, then creates
    scatter plots comparing different wave and vegetation parameters.

    Parameters
    ----------
    df_centroids : pandas.DataFrame
        DataFrame containing centroid data with wave and vegetation parameters.
        Expected columns: 'Hs', 'Hs_L0', 'Wv', 'hv', 'Nv'
    df_lhs_data : pandas.DataFrame
        DataFrame containing Latin Hypercube Sampling data with the same parameters.
        Expected columns: 'Hs', 'Hs_L0', 'Wv', 'hv', 'Nv'
    scatter_points_thick : int

    Returns
    -------
    tuple
        A tuple containing (fig, axes) from the triangular scatter plot

    Notes
    -----
    The peak wave period is calculated using the formula:
    Tp = sqrt((Hs * 2 * π) / (g * Hs_L0))

    Where:
    - Hs: Significant wave height (m)
    - Hs_L0: Wave steepness (dimensionless)
    - g: Gravitational acceleration (9.806 m/s²)

    The function creates scatter plots with:
    - Blue points: LHS data
    - Red points: Centroid data
    """

    # Note: The following lines should use the function parameters instead of 'mda'

    df_centroids["Tp"] = np.sqrt(
        (df_centroids["Hs"].values * 2 * np.pi) / (9.806 * df_centroids["Hs_L0"])
    )
    df_lhs_data["Tp"] = np.sqrt(
        (df_lhs_data["Hs"].values * 2 * np.pi) / (9.806 * df_lhs_data["Hs_L0"])
    )
    df_centroids = df_centroids.drop(columns=["Hs_L0"])
    df_lhs_data = df_lhs_data.drop(columns=["Hs_L0"])
    df_centroids = df_centroids[["Hs", "Tp", "Wv", "hv", "Nv"]]
    df_lhs_data = df_lhs_data[["Hs", "Tp", "Wv", "hv", "Nv"]]

    fig, axes = plot_scatters_in_triangle(
        dataframes=[df_lhs_data, df_centroids],
        s=scatter_points_thick,
        data_colors=["blue", "red"],
    )
    fig.set_size_inches(8, 8)

