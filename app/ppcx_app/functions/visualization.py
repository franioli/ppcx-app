import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image as PILImage

# ================ Plotting Functions with matplotlib ================


def plot_dic_vectors(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    magnitudes: np.ndarray,
    background_image: np.ndarray | PILImage.Image | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    scale: float | None = None,
    scale_units: str = "xy",
    width: float = 0.003,
    headwidth: float = 2.5,
    quiver_alpha: float = 1,
    image_alpha: float = 0.7,
    cmap_name: str = "viridis",
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, object]:
    """Plot DIC displacement vectors using numpy arrays."""
    # Input validation
    arrays = [x, y, u, v, magnitudes]
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("Input arrays must have the same length")

    if len(x) == 0:
        raise ValueError("Input arrays are empty")

    # Set up color normalization
    max_magnitude = vmax if vmax is not None else np.max(magnitudes)
    norm = Normalize(vmin=vmin, vmax=max_magnitude)

    # Set up figure and axes
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display background image if provided
    if background_image is not None:
        ax.imshow(background_image, alpha=image_alpha)

    # Create quiver plot
    q = ax.quiver(
        x,
        y,
        u,
        v,
        magnitudes,
        scale=scale,
        scale_units=scale_units,
        angles="xy",
        cmap=cmap_name,
        norm=norm,
        width=width,
        headwidth=headwidth,
        alpha=quiver_alpha,
    )

    # Add colorbar
    cbar = fig.colorbar(q, ax=ax)
    cbar.set_label("Displacement Magnitude (pixels)")

    # Set title and labels
    if title:
        ax.set_title(title)

    # Disable axis grid and labels
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return fig, ax, q


def plot_dic_scatter(
    x: np.ndarray,
    y: np.ndarray,
    magnitudes: np.ndarray,
    background_image: np.ndarray | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap_name: str = "viridis",
    s: float = 20,
    alpha: float = 0.8,
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, object]:
    """Plot DIC displacement data as a scatter plot colored by magnitude."""
    # Input validation
    if len(x) != len(y) or len(x) != len(magnitudes):
        raise ValueError("Input arrays must have the same length")

    if len(x) == 0:
        raise ValueError("Input arrays are empty")

    # Set up figure and axes
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display background image if provided
    if background_image is not None:
        ax.imshow(background_image, alpha=0.7)

    # Create scatter plot
    scatter = ax.scatter(
        x,
        y,
        c=magnitudes,
        cmap=cmap_name,
        s=s,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax or np.max(magnitudes),
    )

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Displacement Magnitude (pixels)")

    # Set title and labels
    if title:
        ax.set_title(title)

    # Disable axis grid and labels
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return fig, ax, scatter


# ================ Plotting Functions with OpenCV ================


def draw_quiver_on_image_cv2(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    magnitudes: np.ndarray,
    colormap_name: str = "viridis",
    arrow_length_scale: float = 1.0,
    arrow_thickness: int = 1,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Draw arrows directly on the image using OpenCV. Returns a BGR image (uint8)
    ready to save or serve. Coordinates in pixel space (x columns, y rows).
    - arrow_length_scale: multiplier to control size of arrows (tune for your images)
    - magnitudes used to map color via matplotlib colormap
    """
    # ensure uint8 BGR image
    out = image.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    # get colormap mapping from magnitudes -> RGB 0-255
    cmap = cm.get_cmap(colormap_name)
    mag_norm = (magnitudes - magnitudes.min()) / max(
        1e-12, (magnitudes.max() - magnitudes.min())
    )
    colors_rgba = cmap(mag_norm)  # Nx4 in 0..1
    colors_bgr = (colors_rgba[:, :3] * 255)[:, ::-1].astype(
        np.uint8
    )  # convert RGB->BGR

    # draw arrows
    for xi, yi, ui, vi, col in zip(
        x.astype(int), y.astype(int), u, v, colors_bgr, strict=False
    ):
        # end point in pixel coordinates; multiply vector by scale
        end_x = int(round(xi + ui * arrow_length_scale))
        end_y = int(round(yi + vi * arrow_length_scale))
        cv2.arrowedLine(
            out,
            (int(xi), int(yi)),
            (end_x, end_y),
            color=tuple(int(c) for c in col),
            thickness=arrow_thickness,
            tipLength=0.3,
        )
    # optional alpha blending (if you want to overlay arrows on a transparent canvas)
    return out
