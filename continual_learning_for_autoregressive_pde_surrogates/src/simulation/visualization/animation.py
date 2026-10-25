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
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_ylim(y_lim_min, y_lim_max)

    def update(i):
        for j in range(len(list_of_framelists)):
            all_lines[j].set_ydata(list_of_framelists[j][i])
        return all_lines

    ani = animation.FuncAnimation(
        fig, update, frames=len(list_of_framelists[0]), blit=True, interval=49
    )

    fig.tight_layout()

    if file_path:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="me"), bitrate=bitrate)
        ani.save(file_path, writer=writer)

    plt.close(fig)
    return ani


def create_animation_2d(
    frames: List[np.ndarray],
    file_path: str = None,
    colormap: str = "gray",
    fps: int = 15,
    bitrate: int = 1800,
    vmin: float = None,
    vmax: float = None,
):
    sns.set_theme(style="white")
    fig = plt.figure()
    ims = []
    for f in frames:
        im = plt.imshow(f, cmap=colormap, animated=True, vmin=vmin, vmax=vmax)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True)
    if file_path:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="me"), bitrate=bitrate)
        ani.save(file_path, writer=writer)

    plt.close(fig)
    return ani


def create_vector_field_animation_2d(
    frames,
    file_path=None,
    skip=3,
    fps=15,
    scale=None,
    bitrate=1800,
    figsize=(6, 6),
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
):
    sns.set_theme(style="white")

    frames = np.asarray(frames)
    T, _, Ny, Nx = frames.shape

    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    Xs, Ys = X[::skip, ::skip], Y[::skip, ::skip]

    mags = np.sqrt(frames[:, 0] ** 2 + frames[:, 1] ** 2)

    if scale is None:
        max_speed = np.max(mags)
        scale = max_speed * 10.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    u0, v0 = frames[0]
    mag0 = np.sqrt(u0**2 + v0**2)
    im = ax.imshow(mag0, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, animated=True)
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

    def update(i):
        u, v = frames[i]
        mag = np.sqrt(u**2 + v**2)
        im.set_array(mag)
        qv.set_UVC(u[::skip, ::skip], v[::skip, ::skip])
        return im, qv

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        blit=True,
        interval=1000 / fps,
    )

    if file_path:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="me"), bitrate=bitrate)
        ani.save(file_path, writer=writer)

    plt.close(fig)
    return ani
