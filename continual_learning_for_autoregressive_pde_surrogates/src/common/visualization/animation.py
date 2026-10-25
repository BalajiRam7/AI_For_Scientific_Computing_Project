from typing import List, Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_animation_1d(
    list_of_framelists,
    list_of_labels,
    file_path=None,
    y_label="y",
    x_label="x",
    y_lim_min=0,
    y_lim_max=1,
    fps=14,
    bitrate=1800,
    colors=["blue", "green"],
    fig_size_x=4,
    fig_size_y=2,
    fontsize=12,
):
    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
    x = np.arange(len(list_of_framelists[0][0]))
    all_lines = []
    for i in range(len(list_of_framelists)):
        (line,) = ax.plot(
            x,
            list_of_framelists[i][0],
            label=list_of_labels[i],
            color=colors[i],
            linewidth=5,
            alpha=0.8,
        )
        all_lines.append(line)

    ax.legend()
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.legend(fontsize=fontsize)

    def update(i):
        for j in range(len(list_of_framelists)):
            all_lines[j].set_ydata(list_of_framelists[j][i])
        return all_lines

    ani = animation.FuncAnimation(
        fig, update, frames=len(list_of_framelists[0]), blit=True, interval=49
    )

    # Tighten the layout to ensure labels are shown
    fig.tight_layout()

    if file_path:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="me"), bitrate=bitrate)
        ani.save(file_path, writer=writer)

    plt.close(fig)
    return ani

def create_animation_2d(
    frames: List[np.ndarray] | List[List[np.ndarray]],
    file_path: Optional[str] = None,
    colormap: str = "gray",
    fps: int = 15,
    bitrate: int = 1800,
    titles: Optional[List[str]] = None,
):
    sns.set_theme(style="white")

    # --- Normalize input: always work with list-of-lists ---
    if isinstance(frames[0], np.ndarray):
        frames = [frames]

    num_anims = len(frames)
    num_frames = len(frames[0])
    for fset in frames:
        assert len(fset) == num_frames, (
            "All animations must have same number of frames."
        )

    # --- Determine global color scale ---
    vmin = min(np.min(f) for fset in frames for f in fset)
    vmax = max(np.max(f) for fset in frames for f in fset)

    # --- Create figure with N side-by-side subplots ---
    fig, axes = plt.subplots(1, num_anims, figsize=(4 * num_anims, 4))
    if num_anims == 1:
        axes = [axes]

    ims = []
    for i in range(num_frames):
        artists = []
        for j, ax in enumerate(axes):
            f = frames[j][i]
            im = ax.imshow(f, cmap=colormap, vmin=vmin, vmax=vmax, animated=True)
            if titles and j < len(titles):
                ax.set_title(titles[j])
            ax.axis("off")
            artists.append(im)
        ims.append(artists)

    # --- Build animation ---
    ani = animation.ArtistAnimation(fig, ims, interval=1000 / fps, blit=True)

    # --- Save if requested ---
    if file_path:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="me"), bitrate=bitrate)
        ani.save(file_path, writer=writer)

    plt.close(fig)
    return ani


def create_vector_field_animation_2d(
    frames: List[np.ndarray] | List[List[np.ndarray]],
    file_path: Optional[str] = None,
    skip: int = 3,  # subsampling step for quiver density
    fps: int = 15,  # playback and video fps
    scale: Optional[float] = None,  # arrow scaling; auto if None
    bitrate: int = 1800,
    figsize_scale: float = 4.0,  # horizontal size per subplot
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    titles: Optional[List[str]] = None,
):
    """
    Create one or multiple side-by-side 2D vector field animations.

    Each frame must be of shape (2, Ny, Nx): [u, v] components.
    The color indicates the magnitude (speed) at each grid position.
    """

    sns.set_theme(style="white")

    # --- Normalize input ---
    if isinstance(frames[0], np.ndarray):
        frames = [frames]
    num_anims = len(frames)
    num_frames = len(frames[0])
    for fset in frames:
        assert len(fset) == num_frames, (
            "All animations must have the same number of frames."
        )

    # --- Grid setup ---
    Ny, Nx = frames[0][0].shape[1:]
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    Xs, Ys = X[::skip, ::skip], Y[::skip, ::skip]

    # --- Compute magnitude statistics only for these animations ---
    mags_all = []
    for fset in frames:
        for f in fset:
            mags_all.append(np.sqrt(f[0] ** 2 + f[1] ** 2))
    mags_all = np.array(mags_all)

    if vmin is None:
        vmin = float(np.min(mags_all))
    if vmax is None:
        vmax = float(np.max(mags_all))

    # --- Determine arrow scale ---
    if scale is None:
        max_speed = float(np.max(mags_all))
        scale = max_speed * 10.0

    # --- Create subplots ---
    fig, axes = plt.subplots(
        1, num_anims, figsize=(figsize_scale * num_anims, figsize_scale)
    )
    if num_anims == 1:
        axes = [axes]

    ims, quivers = [], []

    # --- Initialize first frame artists ---
    for j, ax in enumerate(axes):
        u0, v0 = frames[j][0]
        mag0 = np.sqrt(u0**2 + v0**2)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if titles and j < len(titles):
            ax.set_title(titles[j])

        im = ax.imshow(
            mag0, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, animated=True
        )
        qv = ax.quiver(
            Xs,
            Ys,
            u0[::skip, ::skip],
            v0[::skip, ::skip],
            color="black",
            scale=scale,
            width=0.002,
            animated=True,
        )

        ims.append(im)
        quivers.append(qv)

    # --- Update function ---
    def update(i):
        artists = []
        for j in range(num_anims):
            u, v = frames[j][i]
            mag = np.sqrt(u**2 + v**2)
            ims[j].set_array(mag)
            ims[j].changed()  # <-- ensures color updates properly when blitting
            quivers[j].set_UVC(u[::skip, ::skip], v[::skip, ::skip])
            artists += [ims[j], quivers[j]]
        return artists

    # --- Create the animation ---
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, blit=True, interval=1000 / fps
    )

    # --- Optional colorbar (shared) ---
    cbar = fig.colorbar(
        ims[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.04
    )
    cbar.set_label("Speed (‖u,v‖)")

    # --- Save if requested ---
    if file_path:
        print(f"Saving animation to {file_path} ...")
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="me"), bitrate=bitrate)
        ani.save(file_path, writer=writer)
        print("Animation saved successfully!")

    plt.close(fig)
    return ani
