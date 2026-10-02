
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


def plot_pca_eofs_pcs(pca, dep_hr, var_to_predict, inputs, pairs=None, n_components=None, figsize=None, dpi=150):
    """Plot the first `n_components` EOF spatial patterns and their standardized PCs in the input space."""

    n_components = min(n_components or len(pca.eofs.n_component), len(pca.eofs.n_component))
    dep = dep_hr["band_data"].squeeze()

    # Default: consecutive input pairs, so each variable appears once
    columns = list(inputs.columns)
    pairs = pairs or [tuple(columns[j:j + 2]) for j in range(0, len(columns) - 1, 2)]
    n_pairs = len(pairs)
    figsize = figsize or (4 + 3 * n_pairs, 3 * n_components)

    fig, axs = plt.subplots(
        n_components, n_pairs + 1, figsize=figsize, dpi=dpi,
        gridspec_kw={"width_ratios": [1.5] + [1] * n_pairs}, squeeze=False, layout="constrained"
    )

    for i, component in enumerate(pca.eofs.n_component.values[:n_components]):
        eof = pca.eofs.sel(n_component=component)[var_to_predict]
        pc = pca.pcs.sel(n_component=component)["PCs"]
        variance = pca.explained_variance_ratio[component].item() * 100

        pc_std = pc.std().item()
        eof_plot, pc_plot = eof * pc_std, pc.values / pc_std
        vmax = abs(eof_plot).max().item()
        pc_vmax = abs(pc_plot).max()

        im = eof_plot.plot(
            ax=axs[i, 0], x="x", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, add_colorbar=False
        )
        axs[i, 0].contour(dep.x, dep.y, dep, levels=[0], colors="k", linewidths=.7)
        fig.colorbar(im, ax=axs[i, 0], label="Flood depth [m]")
        axs[i, 0].set_title(f"EOF {component + 1} ({variance:.1f}%)", fontweight="bold")
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 0].set(xlabel="", ylabel="")

        for j, (x_var, y_var) in enumerate(pairs):
            ax = axs[i, j + 1]
            sc = ax.scatter(
                inputs[x_var], inputs[y_var], c=pc_plot, cmap="RdBu_r",
                vmin=-pc_vmax, vmax=pc_vmax, s=55, edgecolor="k", linewidth=.4, zorder=5
            )
            ax.set(xlabel=x_var, ylabel=y_var)
            ax.grid(lw=.5, alpha=.5)
            if j == 0:
                ax.set_title(f"PC {component + 1}", fontweight="bold", loc="left")

        fig.colorbar(sc, ax=axs[i, 1:], label="Standardized PC")

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


def plot_historical_forcing(sea_level, waves, precipitation, events, start, end):
    """Plot the hourly drivers and the daily events built from them."""

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, dpi=130, layout="constrained")
    period = slice(start, end)
    ev = events.loc[period]

    axes[0].plot(sea_level.loc[period], color="#053B71", lw=0.7, label="Hourly sea level")
    axes[0].plot(ev.index + pd.Timedelta(hours=12), ev["SL"], "o", ms=3, color="#E8590C", label="Daily maximum (event)")
    axes[0].set_ylabel("Sea level [m NAVD88]")

    axes[1].plot(waves.loc[period, "Hs"], color="#053B71", lw=0.7, label="Hourly Hs")
    axes[1].plot(ev.index + pd.Timedelta(hours=12), ev["Hs"], "o", ms=3, color="#E8590C", label="Hs at high tide (event)")
    axes[1].set_ylabel("Offshore Hs [m]")

    axes[2].bar(precipitation.loc[period].index, precipitation.loc[period], width=1 / 24, color="#5AA1D4", label="Hourly rain")
    ax2 = axes[2].twinx()
    ax2.plot(ev.index + pd.Timedelta(hours=12), ev["precipitation"], "o", ms=3, color="#E8590C", label="Daily total (event)")
    axes[2].set_ylabel("Rain [mm/h]")
    ax2.set_ylabel("Daily rain [mm]", color="#E8590C")

    for ax in axes:
        ax.grid(lw=0.4)
        ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    return fig, axes


def plot_events_vs_training(events, training, pairs=(("Hs", "Tp"), ("SL", "Hs"), ("precipitation", "precip_duration"))):
    """Scatter of the historical events over the training scenarios of the surrogate."""

    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6), dpi=130, layout="constrained")

    for ax, (x, y) in zip(axes, pairs):
        ax.scatter(events[x], events[y], s=2, alpha=0.3, color="#5AA1D4", label="Daily events")
        ax.scatter(training[x], training[y], s=18, color="#E8590C", edgecolor="k", lw=0.4, label="SFINCS scenarios")
        ax.set(xlabel=x, ylabel=y)
        ax.grid(lw=0.4)

    axes[0].legend(fontsize=8)

    return fig, axes


def plot_flood_statistics(maps, dep_hr, titles, labels, cmaps, vmax=None, log=None, figsize=(13, 4.2)):
    """Plot several flood statistic maps side by side, each with its own colorbar."""

    from matplotlib.colors import LogNorm

    vmax = vmax or [None] * len(maps)
    log = log or [False] * len(maps)
    fig, axes = plt.subplots(1, len(maps), figsize=figsize, sharex=True, sharey=True, dpi=160, layout="constrained")
    axes = np.atleast_1d(axes)

    dep = dep_hr["band_data"].squeeze()

    for ax, da, title, label, cmap, vm, lg in zip(axes, maps, titles, labels, cmaps, vmax, log):
        scale = {"norm": LogNorm(vmin=1, vmax=vm)} if lg else {"vmin": 0, "vmax": vm}
        da.where(da > 0).plot(ax=ax, cmap=cmap, **scale, cbar_kwargs={"label": label, "shrink": 0.8, "orientation": "horizontal"})
        ax.contour(dep.x, dep.y, dep, levels=[0], colors="k", linewidths=.7)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")

    return fig, axes
