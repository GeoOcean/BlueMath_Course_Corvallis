"""Plot annual DWT probabilities, PCA modes and AWT clusters.

Inputs are passed explicitly; functions return Matplotlib figures/axes without
calling show(). Probability tables use fractions; anomaly tables in
plot_annual_anomalies use percentage points, as computed in the notebook.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D

def plot_annual_probabilities(annual_probability):
    """Return pages of annual probability maps; input probabilities are fractions."""
    years = annual_probability.index
    dwt_ids = annual_probability.columns
    probabilities = annual_probability * 100
    n_dwt = len(dwt_ids)
    grid_cols = int(np.ceil(np.sqrt(n_dwt)))
    grid_rows = int(np.ceil(n_dwt / grid_cols))
    plots_per_page = 49
    panel_cols = 7
    vmax = float(np.nanmax(probabilities.to_numpy()))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("lightgrey")
    annual_figures = []

    for start in range(0, len(years), plots_per_page):
        page_years = years[start:start + plots_per_page]
        panel_rows = 7
        fig, axes = plt.subplots(panel_rows, panel_cols, figsize=(15, 15),
                                 squeeze=False, layout="constrained")
        for ax, year in zip(axes.flat, page_years):
            values = np.full(grid_rows * grid_cols, np.nan)
            values[:n_dwt] = probabilities.loc[year].to_numpy()
            values = values.reshape(grid_rows, grid_cols)
            mesh = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax,
                             origin="upper", interpolation="nearest")
            for pos, dwt in enumerate(dwt_ids):
                row, col = divmod(pos, grid_cols)
                color = "white" if values[row, col] > 0.55 * vmax else "0.3"
                ax.text(col, row, str(dwt), ha="center", va="center", fontsize=8, color=color)
            ax.set_title(str(year))
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in list(axes.flat)[len(page_years):]:
            ax.set_visible(False)
        fig.colorbar(mesh, ax=list(axes.flat)[:len(page_years)], shrink=0.85,
                     pad=0.02, label="Annual probability (%)")
        fig.suptitle("DWT probabilities by year · cell numbers identify DWTs")
        annual_figures.append(fig)
    return annual_figures


def plot_annual_anomalies(annual_anomaly):
    """Return pages of annual anomaly maps; input values are percentage points."""
    years = annual_anomaly.index
    dwt_ids = annual_anomaly.columns
    probabilities = annual_anomaly
    n_dwt = len(dwt_ids)
    grid_cols = int(np.ceil(np.sqrt(n_dwt)))
    grid_rows = int(np.ceil(n_dwt / grid_cols))
    plots_per_page = 49
    panel_cols = 7
    vmax = max(float(np.nanmax(np.abs(probabilities.to_numpy()))), 1e-12)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")
    anomaly_figures = []

    for start in range(0, len(years), plots_per_page):
        page_years = years[start:start + plots_per_page]
        panel_rows = 7
        fig, axes = plt.subplots(panel_rows, panel_cols, figsize=(15, 15),
                                 squeeze=False, layout="constrained")
        for ax, year in zip(axes.flat, page_years):
            values = np.full(grid_rows * grid_cols, np.nan)
            values[:n_dwt] = probabilities.loc[year].to_numpy()
            values = values.reshape(grid_rows, grid_cols)
            mesh = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax,
                             origin="upper", interpolation="nearest")
            for pos, dwt in enumerate(dwt_ids):
                row, col = divmod(pos, grid_cols)
                color = "white" if abs(values[row, col]) > 0.55 * vmax else "0.3"
                ax.text(col, row, str(dwt), ha="center", va="center", fontsize=8, color=color)
            ax.set_title(str(year))
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in list(axes.flat)[len(page_years):]:
            ax.set_visible(False)
        fig.colorbar(mesh, ax=list(axes.flat)[:len(page_years)], shrink=0.85,
                     pad=0.02, label="Probability anomaly (percentage points)")
        fig.suptitle(f"Annual DWT anomalies relative to {years.min()}–{years.max()}")
        anomaly_figures.append(fig)
    return anomaly_figures


def _mode_panels(n_modes, grid_rows, grid_cols, *, with_legend=False):
    """Allocate equal-height map/series panels and separate annotation rows."""
    map_aspect = grid_rows / grid_cols
    fig = plt.figure(figsize=(12, n_modes * (1.25 * map_aspect + 0.5)
                              + 0.9 + (0.45 if with_legend else 0)),
                     layout="constrained")
    offset = int(with_legend)
    heights = ([0.35] if with_legend else []) + [1.25 * map_aspect] * n_modes + [0.12]
    grid = fig.add_gridspec(n_modes + offset + 1, 2,
                           width_ratios=[1, 8], height_ratios=heights)
    axes = np.empty((n_modes, 2), dtype=object)
    for row in range(n_modes):
        for col in range(2):
            ax = fig.add_subplot(grid[row + offset, col])
            ax.set_box_aspect(map_aspect / (1 if col == 0 else 8))
            axes[row, col] = ax
    colorbar_ax = fig.add_subplot(grid[-1, 0])
    legend_ax = fig.add_subplot(grid[0, :]) if with_legend else None
    if legend_ax is not None:
        legend_ax.set_axis_off()
    return fig, axes, colorbar_ax, legend_ax


def plot_pca_modes(annual_dwt_pcs, annual_dwt_eofs, explained_variance_ratio, dwt_ids):
    """Return EOF maps and annual PC curves with explained variance."""
    grid_cols = int(np.ceil(np.sqrt(len(dwt_ids))))
    grid_rows = int(np.ceil(len(dwt_ids) / grid_cols))
    n_modes = annual_dwt_pcs.sizes["n_component"]
    fig_pca, axes_pca, colorbar_ax, _ = _mode_panels(n_modes, grid_rows, grid_cols)
    eof_limit = float(np.abs(annual_dwt_eofs.values).max())
    eof_norm = TwoSlopeNorm(vmin=-eof_limit, vcenter=0, vmax=eof_limit)
    pc_years = annual_dwt_pcs.year.values

    for mode, (ax_map, ax_pc) in enumerate(axes_pca):
        pattern = np.full(grid_rows * grid_cols, np.nan)
        pattern[:len(dwt_ids)] = annual_dwt_eofs.isel(n_component=mode).values
        eof_image = ax_map.imshow(pattern.reshape(grid_rows, grid_cols),
                                  cmap="RdBu_r", norm=eof_norm, interpolation="nearest")
        ax_map.set(xticks=[], yticks=[])
        variance = explained_variance_ratio[mode] * 100
        ax_map.set_title(f"EOF {mode + 1} · {variance:.1f}%", fontsize=10)

        scores = annual_dwt_pcs.PCs.isel(n_component=mode).values
        ax_pc.plot(pc_years, scores, color="black", linewidth=1.2)
        ax_pc.axhline(0, color="0.5", linewidth=0.6)
        ax_pc.set_ylabel(f"PC {mode + 1} (pp)")
        ax_pc.grid(alpha=0.4)
        ax_pc.set_xlim(pc_years.min() - 1, pc_years.max() + 1)
        ax_pc.tick_params(labelbottom=mode == n_modes - 1)
    axes_pca[-1, 1].set_xlabel("Year")
    fig_pca.colorbar(eof_image, cax=colorbar_ax, orientation="horizontal", label="EOF loading")
    fig_pca.suptitle("PCA of annual DWT probability anomalies")
    return fig_pca, axes_pca


def plot_year_clusters(annual_cluster_input, annual_dwt_eofs, explained_variance_ratio, dwt_ids, year_clusters, cluster_colors):
    """Return EOF maps and PC curves colored by the supplied year clusters."""
    grid_cols = int(np.ceil(np.sqrt(len(dwt_ids))))
    grid_rows = int(np.ceil(len(dwt_ids) / grid_cols))
    year_clusters = year_clusters.reindex(annual_cluster_input.index)
    if year_clusters.isna().any():
        raise ValueError("Every plotted year must have a cluster assignment.")
    mode_count = annual_cluster_input.shape[1]
    fig_year_clusters, axes_year_clusters, colorbar_ax, legend_ax = _mode_panels(
        mode_count, grid_rows, grid_cols, with_legend=True,
    )
    eof_limit = float(np.abs(annual_dwt_eofs.values).max())
    eof_norm = TwoSlopeNorm(vmin=-eof_limit, vcenter=0, vmax=eof_limit)
    plot_years = annual_cluster_input.index.to_numpy()
    point_colors = year_clusters.map(cluster_colors).tolist()

    for mode, (ax_map, ax_pc) in enumerate(axes_year_clusters):
        pattern = np.full(grid_rows * grid_cols, np.nan)
        pattern[:len(dwt_ids)] = annual_dwt_eofs.isel(n_component=mode).values
        eof_image = ax_map.imshow(pattern.reshape(grid_rows, grid_cols),
                                  cmap="RdBu_r", norm=eof_norm, interpolation="nearest")
        ax_map.set(xticks=[], yticks=[])
        variance = explained_variance_ratio[mode] * 100
        ax_map.set_title(f"EOF {mode + 1} · {variance:.1f}%", fontsize=10)
        scores = annual_cluster_input.iloc[:, mode]
        ax_pc.plot(plot_years, scores, color="black", linewidth=1.2)
        ax_pc.scatter(plot_years, scores, c=point_colors, s=30, zorder=3)
        ax_pc.set_ylabel(f"PC {mode + 1}")

    for row, ax in enumerate(axes_year_clusters[:, 1]):
        ax.axhline(0, color="0.6", linewidth=0.5)
        ax.grid(alpha=0.35)
        ax.set_xlim(plot_years.min() - 1, plot_years.max() + 1)
        ax.tick_params(labelbottom=row == mode_count - 1)
    axes_year_clusters[-1, 1].set_xlabel("Year")
    legend_handles = [Line2D([], [], marker="o", linestyle="none", color=cluster_colors[k],
                             label=f"Cluster {k} (n={int((year_clusters == k).sum())})")
                      for k in cluster_colors]
    legend_ax.legend(handles=legend_handles, ncol=min(5, len(legend_handles)),
                     loc="center", fontsize=9)
    fig_year_clusters.colorbar(eof_image, cax=colorbar_ax,
                               orientation="horizontal", label="EOF loading")
    fig_year_clusters.suptitle(
        f"{year_clusters.nunique()} clusters of years from DWT PCs · consistent year colors across panels")
    return fig_year_clusters, axes_year_clusters


def plot_pc_pairs(annual_cluster_input, year_clusters, cluster_colors, hull_padding=0.04):
    """Return pairwise PC scatter plots with buffered cluster hulls."""
    from shapely.geometry import MultiPoint

    # Every unique PC pair, with one point per year and consistent cluster colors.
    scatter_pcs = annual_cluster_input
    scatter_clusters = year_clusters.reindex(scatter_pcs.index)
    if scatter_clusters.isna().any():
        raise ValueError("Every plotted year must have a cluster assignment.")
    n_scatter_pcs = scatter_pcs.shape[1]
    if n_scatter_pcs < 2:
        raise ValueError("At least two PCs are needed for a pairwise scatter plot.")
    fig_pc_pairs, axes_pc_pairs = plt.subplots(
        n_scatter_pcs - 1, n_scatter_pcs - 1,
        figsize=(3 * (n_scatter_pcs - 1), 3 * (n_scatter_pcs - 1)),
        squeeze=False, layout="constrained",
    )
    pc_limits = {}
    for column in scatter_pcs:
        low, high = scatter_pcs[column].min(), scatter_pcs[column].max()
        margin = 0.08 * (high - low) if high > low else 1
        pc_limits[column] = (low - margin, high + margin)

    for row in range(n_scatter_pcs - 1):
        for col in range(n_scatter_pcs - 1):
            ax = axes_pc_pairs[row, col]
            if col < row:
                ax.set_visible(False)
                continue
            x_name, y_name = scatter_pcs.columns[col + 1], scatter_pcs.columns[row]
            # Buffer the convex hull in normalized coordinates for rounded corners.
            # A positive buffer keeps every observed point inside the envelope.
            scale = np.ptp(scatter_pcs[[x_name, y_name]].to_numpy(), axis=0)
            scale = np.where(scale > 0, scale, 1.0)
            for cluster, color in cluster_colors.items():
                points = scatter_pcs.loc[scatter_clusters == cluster, [x_name, y_name]].to_numpy()
                points = np.unique(points[np.isfinite(points).all(axis=1)], axis=0)
                if len(points) == 0:
                    continue
                envelope = MultiPoint(points / scale).convex_hull.buffer(
                    hull_padding, resolution=32, join_style=1,
                )
                boundary = np.asarray(envelope.exterior.coords) * scale
                ax.fill(boundary[:, 0], boundary[:, 1],
                        facecolor=to_rgba(color, 0.10), edgecolor=to_rgba(color, 0.8),
                        linewidth=.5, zorder=1)
            ax.scatter(scatter_pcs[x_name], scatter_pcs[y_name],
                       c=scatter_clusters.map(cluster_colors).tolist(), s=50,
                       alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
            ax.axhline(0, color="0.7", linewidth=0.6, zorder=0)
            ax.axvline(0, color="0.7", linewidth=0.6, zorder=0)
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
            ax.set(xlabel=f"{x_name} (pp)", ylabel=f"{y_name} (pp)",
                   xlim=pc_limits[x_name], ylim=pc_limits[y_name])
    handles = [Line2D([], [], marker="o", linestyle="none", color=cluster_colors[k],
                      label=f"Cluster {k} (n={int((scatter_clusters == k).sum())})")
               for k in cluster_colors]
    fig_pc_pairs.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.04, 0.55),
                         frameon=True, fontsize=11)
    fig_pc_pairs.suptitle("Annual DWT PCs · one point per year, colored by cluster")
    return fig_pc_pairs, axes_pc_pairs


def plot_awt_centroids(awt_probability_centroids, dwt_ids, awt_year_counts, cluster_colors):
    """Return mean probability maps by AWT; input probabilities are fractions."""
    n_awts = len(awt_probability_centroids)
    fig_awt_centroids, axes_awt_centroids = plt.subplots(
        1, n_awts, figsize=(2 * n_awts, 2.6), squeeze=False, layout="constrained",
    )
    centroid_vmax = float(awt_probability_centroids.to_numpy().max() * 100)
    centroid_cols = int(np.ceil(np.sqrt(len(dwt_ids))))
    centroid_rows = int(np.ceil(len(dwt_ids) / centroid_cols))
    for ax, (awt, probabilities) in zip(axes_awt_centroids.flat, awt_probability_centroids.iterrows()):
        field = np.full(centroid_rows * centroid_cols, np.nan)
        field[:len(dwt_ids)] = probabilities.reindex(dwt_ids).to_numpy() * 100
        field = field.reshape(centroid_rows, centroid_cols)
        centroid_image = ax.imshow(field, cmap="Blues", vmin=0, vmax=centroid_vmax,
                                   origin="upper", interpolation="nearest")
        for pos, dwt in enumerate(dwt_ids):
            row, col = divmod(pos, centroid_cols)
            ax.text(col, row, str(dwt), ha="center", va="center", fontsize=8,
                    color="white" if field[row, col] > 0.55 * centroid_vmax else "0.3")
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"AWT {awt} · {int(awt_year_counts.loc[awt])} years", color=cluster_colors[awt])
        for spine in ax.spines.values():
            spine.set_color(cluster_colors[awt])
            spine.set_linewidth(2)
    fig_awt_centroids.colorbar(centroid_image, ax=list(axes_awt_centroids.flat),
                              shrink=0.8, pad=0.02, label="Mean annual DWT probability (%)")
    fig_awt_centroids.suptitle("AWT centroids · mean DWT probabilities across member years")
    return fig_awt_centroids, axes_awt_centroids


def plot_awt_anomalies(awt_anomaly_centroids, dwt_ids, awt_year_counts, cluster_colors):
    """Return AWT anomaly maps; input anomalies are differences of fractions."""
    n_awts = len(awt_anomaly_centroids)
    fig_awt_anomalies, axes_awt_anomalies = plt.subplots(
        1, n_awts, figsize=(2 * n_awts, 2.6), squeeze=False, layout="constrained",
    )
    centroid_vmax = max(float(np.abs(awt_anomaly_centroids.to_numpy()).max() * 100), 1e-12)
    centroid_cols = int(np.ceil(np.sqrt(len(dwt_ids))))
    centroid_rows = int(np.ceil(len(dwt_ids) / centroid_cols))
    for ax, (awt, probabilities) in zip(axes_awt_anomalies.flat, awt_anomaly_centroids.iterrows()):
        field = np.full(centroid_rows * centroid_cols, np.nan)
        field[:len(dwt_ids)] = probabilities.reindex(dwt_ids).to_numpy() * 100
        field = field.reshape(centroid_rows, centroid_cols)
        centroid_image = ax.imshow(field, cmap="RdBu_r", vmin=-centroid_vmax, vmax=centroid_vmax,
                                   origin="upper", interpolation="nearest")
        for pos, dwt in enumerate(dwt_ids):
            row, col = divmod(pos, centroid_cols)
            ax.text(col, row, str(dwt), ha="center", va="center", fontsize=8,
                    color="white" if abs(field[row, col]) > 0.55 * centroid_vmax else "0.3")
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"AWT {awt} · {int(awt_year_counts.loc[awt])} years", color=cluster_colors[awt])
        for spine in ax.spines.values():
            spine.set_color(cluster_colors[awt])
            spine.set_linewidth(2)
    fig_awt_anomalies.colorbar(centroid_image, ax=list(axes_awt_anomalies.flat),
                              shrink=0.8, pad=0.02, label="DWT probability anomaly (percentage points)")
    fig_awt_anomalies.suptitle("AWT centroids · anomalies relative to the mean across all years")
    return fig_awt_anomalies, axes_awt_anomalies


