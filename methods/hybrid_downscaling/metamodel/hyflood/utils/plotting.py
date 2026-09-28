
import matplotlib.pyplot as plt

def plot_output_cases(output, dep_hr, figsize=(11, 7), cmap='BuPu',vmin=None, vmax=None):

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True, dpi=160)
    axes = axes.ravel()

    dep = dep_hr["band_data"].squeeze()

    for i, ds in enumerate(output):

        im = ds.zsmax.plot(ax=axes[i], cmap=cmap, add_colorbar=False, vmin=vmin, vmax=vmax)

        axes[i].contour(
            dep.x, dep.y, dep,
            levels=[0], colors="k", linewidths=.7
        )

        axes[i].set_title(f"Case {i+1}")
        axes[i].set_aspect("equal")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    fig.colorbar(im, ax=axes, shrink=0.8, label='Water Depth (m)')

    return fig, axes


def plot_pca_eofs_pcs(pca, dep_hr, var_to_predict, figsize=None, dpi=150):
    """Plot EOF spatial patterns and standardized PCs."""

    n_components = len(pca.eofs.n_component)
    dep = dep_hr["band_data"].squeeze()
    figsize = figsize or (12, 3 * n_components)

    fig, axs = plt.subplots(
        n_components, 2, figsize=figsize, dpi=dpi,
        gridspec_kw={"width_ratios": [1, 1.5]}, squeeze=False
    )

    for i, component in enumerate(pca.eofs.n_component.values):
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
        fig.colorbar(im, ax=axs[i, 0], fraction=.046, pad=.04, label="Water depth [m]")
        axs[i, 0].set_title(f"EOF {component + 1} ({variance:.1f}%)", fontweight="bold")
        axs[i, 0].title.set_fontweight("bold")
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])

        axs[i, 1].plot(pc_plot, lw=1.5, c="#5AA1D4")
        axs[i, 1].scatter(range(len(pc_plot)), pc_plot, c="#053B71", s=55, zorder=5)
        axs[i, 1].axhline(0, color="grey", lw=.5)
        axs[i, 1].set(title=f"PC {component + 1}", ylabel="Standardized PC")
        axs[i, 1].title.set_fontweight("bold")

    plt.tight_layout()
    return fig, axs


def plot_sfincs_forcings(slowly_waterlevel_forcing, quickly_waterlevel_forcing, precipitation_forcing):
    """Plot the SFINCS forcings."""

    fig, axes = plt.subplots(3, 2, figsize=(8, 5), sharex=True, sharey=True)
    axes = axes.ravel()

    c_sl, c_p = "#69A9A3", "#C68364"

    for i in range(len(slowly_waterlevel_forcing)):
        ax = axes[i]

        sl = slowly_waterlevel_forcing[i] + quickly_waterlevel_forcing[i]
        p = precipitation_forcing[i]

        t_sl = (sl.index - sl.index[0]).total_seconds() / 3600
        t_p = (p.index - p.index[0]).total_seconds() / 3600

        ax.plot(t_sl, sl, c=c_sl)
        ax.set_ylabel("Sea level [m]", color=c_sl, fontweight='bold')
        ax.tick_params(axis="y", colors=c_sl)

        ax2 = ax.twinx()
        ax2.fill_between(t_p, 0, p.values.squeeze(), color=c_p, alpha=.5)
        ax2.set_ylabel("Rain [mm/h]", color=c_p, fontweight='bold')
        ax2.tick_params(axis="y", colors=c_p)

        ax.set_title(f"Case {i+1}")
        ax.grid(lw=.5)

    for ax in axes[-2:]:
        ax.set_xlabel("Time [h]")
    ax.set_xlim(0, 24)
    plt.tight_layout()
    plt.show()
    return fig, axes