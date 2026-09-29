
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

def plot_output_cases(output, dep_hr, ncols=3, figsize=None, cmap='BuPu', vmin=None, vmax=None, titles=None, label='Max. flood depth [m]'):
    """Plot the maximum flood depth of each simulated case."""

    nrows = int(np.ceil(len(output) / ncols))
    figsize = figsize or (11, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, dpi=160, squeeze=False)
    axes = axes.ravel()

    dep = dep_hr["band_data"].squeeze()

    for i, ds in enumerate(output):

        da = ds.zsmax if isinstance(ds, xr.Dataset) else ds
        im = da.plot(ax=axes[i], cmap=cmap, add_colorbar=False, vmin=vmin, vmax=vmax)

        axes[i].contour(
            dep.x, dep.y, dep,
            levels=[0], colors="k", linewidths=.7
        )

        axes[i].set_title(titles[i] if titles is not None else f"Case {i}", fontsize=9)
        axes[i].set_aspect("equal")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for ax in axes[len(output):]:
        ax.set_visible(False)

    fig.colorbar(im, ax=axes, shrink=0.8, label=label)

    return fig, axes


def plot_pca_eofs_pcs(pca, dep_hr, var_to_predict, n_components=None, figsize=None, dpi=150):
    """Plot the first `n_components` EOF spatial patterns and standardized PCs."""

    n_components = min(n_components or len(pca.eofs.n_component), len(pca.eofs.n_component))
    dep = dep_hr["band_data"].squeeze()
    figsize = figsize or (12, 3 * n_components)

    fig, axs = plt.subplots(
        n_components, 2, figsize=figsize, dpi=dpi,
        gridspec_kw={"width_ratios": [1, 1.5]}, squeeze=False
    )

    for i, component in enumerate(pca.eofs.n_component.values[:n_components]):
        eof = pca.eofs.sel(n_component=component)[var_to_predict]
        pc = pca.pcs.sel(n_component=component)["PCs"]
        variance = pca.explained_variance_ratio[component].item() * 100

        pc_std = pc.std().item()
        eof_plot, pc_plot = eof * pc_std, pc / pc_std
        vmax = abs(eof_plot).max().item()

        im = eof_plot.plot(
            ax=axs[i, 0], x="x", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, add_colorbar=False
        )
        axs[i, 0].contour(dep.x, dep.y, dep, levels=[0], colors="k", linewidths=.7)
        fig.colorbar(im, ax=axs[i, 0], fraction=.046, pad=.04, label="Flood depth [m]")
        axs[i, 0].set_title(f"EOF {component + 1} ({variance:.1f}%)", fontweight="bold")
        axs[i, 0].title.set_fontweight("bold")
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 0].set(xlabel="", ylabel="")

        axs[i, 1].plot(pc_plot, lw=1.5, c="#5AA1D4")
        axs[i, 1].scatter(range(len(pc_plot)), pc_plot, c="#053B71", s=55, zorder=5)
        axs[i, 1].axhline(0, color="grey", lw=.5)
        axs[i, 1].set(title=f"PC {component + 1}", xlabel="Case", ylabel="Standardized PC")
        axs[i, 1].title.set_fontweight("bold")

    plt.tight_layout()
    return fig, axs


def plot_sfincs_forcings(slowly_waterlevel_forcing, quickly_waterlevel_forcing, precipitation_forcing, reference_time, ncols=2):
    """Plot the SFINCS water-level and precipitation forcings of each case."""

    n_cases = len(slowly_waterlevel_forcing)
    nrows = int(np.ceil(n_cases / ncols))
    reference_time = pd.Timestamp(reference_time)

    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 1.8 * nrows), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    c_sl, c_p = "#69A9A3", "#C68364"
    p_max = max(p.values.max() for p in precipitation_forcing)

    for i in range(n_cases):
        ax = axes[i]

        sl = slowly_waterlevel_forcing[i] + quickly_waterlevel_forcing[i]
        p = precipitation_forcing[i]

        # Common time axis (hours from the reference time) for both forcings
        t_sl = (sl.index - reference_time).total_seconds() / 3600
        t_p = (p.index - reference_time).total_seconds() / 3600

        ax.plot(t_sl, sl, c=c_sl, lw=.8)
        ax.set_ylabel("Sea level [m]", color=c_sl, fontweight='bold')
        ax.tick_params(axis="y", colors=c_sl)

        ax2 = ax.twinx()
        ax2.fill_between(t_p, 0, p.values.squeeze(), color=c_p, alpha=.5)
        ax2.set_ylim(0, 1.1 * p_max if p_max > 0 else 1)
        ax2.set_ylabel("Rain [mm/h]", color=c_p, fontweight='bold')
        ax2.tick_params(axis="y", colors=c_p)

        ax.set_title(f"Case {i}")
        ax.grid(lw=.5)

    for ax in axes[n_cases:]:
        ax.set_visible(False)

    for ax in axes[n_cases - ncols:n_cases]:
        ax.set_xlabel("Time [h]")

    plt.tight_layout()
    return fig, axes
