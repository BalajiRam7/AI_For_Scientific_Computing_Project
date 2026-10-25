from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_sample_grid_1d(samples, sample_keys):
    sns.set_theme(style="white")

    n_rows = len(samples)
    n_cols = len(samples[0])

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            sns.lineplot(
                data=samples[i][j],
                x=sample_keys["x"],
                y=sample_keys["y"],
                hue=sample_keys["hue"],
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Keep only the legend for the first plot ont he first row with best position automatically
            if i == 0 and j == 0:
                ax.legend(loc="upper right")
            else:
                ax.legend().remove()

    # Set title for each row and column using row_titles and col_titles
    for j in range(n_cols):
        axes[0, j].set_title(f"Sample {j + 1}")

    fig.tight_layout()
    return fig


def plot_sample_grid_2d(samples, sample_keys, stats):
    sns.set_theme(style="white")

    n_rows = len(samples)
    n_cols = len(samples[0])

    # Create a grid of n_rows x n_cols subplots where each subplot contains num_side_by_side_figures figures side by side
    num_side_by_side_figures = len(sample_keys)
    fig, axes = plt.subplots(
        n_rows,
        n_cols * num_side_by_side_figures,
        figsize=(4 * n_cols * num_side_by_side_figures, 3 * n_rows),
        squeeze=False,
    )

    min_u = stats["u"]["min"]
    max_u = stats["u"]["max"]

    for i in range(n_rows):
        for j in range(n_cols):
            for k, key_name in enumerate(sample_keys):
                ax = axes[i, j * num_side_by_side_figures + k]
                u = samples[i][j][key_name]
                # TODO: manual denomralization
                u = u * (max_u - min_u) + min_u

                ax.imshow(u, vmin=min_u, vmax=max_u)
                ax.set_title(key_name)
                ax.set_xlabel("")
                ax.set_ylabel("")
                # Show colorbar
                cbar = ax.figure.colorbar(ax.imshow(u, vmin=min_u, vmax=max_u), ax=ax)

    fig.tight_layout()
    return fig


def plot_sample_grid_2dvec(samples, sample_keys, stats, skip=3, cmap="coolwarm"):
    sns.set_theme(style="white")

    n_rows = len(samples)
    n_cols = len(samples[0])
    num_side_by_side_figures = len(sample_keys)

    fig, axes = plt.subplots(
        n_rows,
        n_cols * num_side_by_side_figures,
        figsize=(4 * n_cols * num_side_by_side_figures, 3 * n_rows),
        squeeze=False,
    )

    # Get u normalization stat from data
    min_u = stats["u"]["min"]
    max_u = stats["u"]["max"]

    vmin = min_u
    vmax = max_u

    for i in range(n_rows):
        for j in range(n_cols):
            for k, key_name in enumerate(sample_keys):
                ax = axes[i, j * num_side_by_side_figures + k]

                field = samples[i][j][key_name]  # shape (2, H, W)
                u, v = field[0], field[1]

                # TODO: remove manual denormalization
                # Denormalize u and v
                u = u * (max_u - min_u) + min_u
                v = v * (max_u - min_u) + min_u

                H, W = u.shape

                # Compute magnitude
                mag = np.sqrt(u**2 + v**2)

                # Create coordinate grid
                X, Y = np.meshgrid(np.arange(W), np.arange(H))

                # Background color for magnitude
                im = ax.imshow(mag, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

                # Quiver plot (arrows)
                ax.quiver(
                    X[::skip, ::skip],
                    Y[::skip, ::skip],
                    u[::skip, ::skip],
                    v[::skip, ::skip],
                    color="black",
                    scale=None,
                    width=0.002,
                )

                ax.set_title(key_name)
                ax.set_xticks([])
                ax.set_yticks([])

                # Add colorbar for magnitude
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("|v|")

    fig.tight_layout()
    return fig


def plot_bar_using_dict(
    log_dict: Dict[str, Any],
):
    all_labels = log_dict["labels"]
    all_values = log_dict["values"]

    n = len(all_values)
    indices = np.arange(n)

    fig, ax = plt.subplots()
    ax.bar(indices, all_values)
    ax.set_xticks(indices)
    ax.set_xticklabels(all_labels)

    return fig
