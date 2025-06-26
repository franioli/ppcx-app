from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def visualize_dic_dataframe(
    df: pd.DataFrame,
    background_image: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    scale: float | None = None,
    scale_units: str = "xy",
    width: float = 0.003,
    headwidth: float = 2.5,
    alpha: float = 0.8,
    cmap_name: str = "batlow",
    show: bool = False,
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
) -> tuple[Figure, Axes, object] | None:
    """Visualize DIC displacement data from a pandas DataFrame using matplotlib quiver plot.

    Args:
        df: DataFrame containing DIC displacement data with columns:
            'seed_x_px', 'seed_y_px', 'displacement_x_px', 'displacement_y_px',
            'displacement_magnitude_px', 'master_timestamp'
        background_image: Optional background image array to display behind vectors
        output_dir: Directory to save plots, or None to just show
        filename: Custom filename for saved plot (without extension)
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization (if None, uses max magnitude)
        scale: Quiver scale parameter. If None, quiver auto-scales arrows
        scale_units: Units for quiver scaling (default "xy")
        width: Width of the quiver arrows
        headwidth: Headwidth for quiver arrows
        alpha: Alpha transparency for quiver arrows
        cmap_name: Name of colormap (from cmcrameri or matplotlib)
        show: If True, show the plot interactively
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the figure
        ax: Optional matplotlib Axes to plot on
        fig: Optional matplotlib Figure to use

    Returns:
        If ax is provided, returns (fig, ax, quiver_obj) for the plot.
        Otherwise, saves or shows the plot and returns None.

    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    required_columns = [
        "seed_x_px",
        "seed_y_px",
        "displacement_x_px",
        "displacement_y_px",
        "displacement_magnitude_px",
    ]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

    # Handle colormap selection
    cmap = None
    if hasattr(cmc, cmap_name):
        cmap = getattr(cmc, cmap_name)
    elif cmap_name in plt.colormaps():
        cmap = plt.colormaps.get_cmap(cmap_name)
    else:
        print(f"Colormap '{cmap_name}' not found. Falling back to 'viridis'.")
        cmap = plt.colormaps.get_cmap("viridis")

    # Set up color normalization
    max_magnitude = vmax if vmax is not None else df["displacement_magnitude_px"].max()
    norm = Normalize(vmin=vmin, vmax=max_magnitude)

    # Set up figure and axes
    if ax is not None:
        current_ax = ax
        current_fig = fig if fig is not None else ax.figure
    else:
        current_fig, current_ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display background image if provided
    if background_image is not None:
        current_ax.imshow(background_image, alpha=0.7)

    # Extract data for quiver plot
    X = df["seed_x_px"].values
    Y = df["seed_y_px"].values
    U = df["displacement_x_px"].values
    V = df["displacement_y_px"].values
    C = df["displacement_magnitude_px"].values

    # Set up quiver plot parameters
    quiver_kwargs = {
        "angles": "xy",
        "scale_units": scale_units,
        "cmap": cmap,
        "norm": norm,
        "width": width,
        "headwidth": headwidth,
        "alpha": alpha,
    }
    if scale is not None:
        quiver_kwargs["scale"] = scale

    # Create quiver plot
    q = current_ax.quiver(X, Y, U, V, C, **quiver_kwargs)

    # Add colorbar
    cbar = current_fig.colorbar(q, ax=current_ax)
    cbar.set_label("Displacement Magnitude (pixels)")

    # Set title and labels
    if "master_timestamp" in df.columns:
        timestamp = df["master_timestamp"].iloc[0]
        if isinstance(timestamp, str):
            time_str = timestamp
        else:
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        current_ax.set_title(f"DIC Displacement Vectors - {time_str}")
    else:
        current_ax.set_title("DIC Displacement Vectors")

    current_ax.set_xlabel("X (pixels)")
    current_ax.set_ylabel("Y (pixels)")

    # Handle output
    if ax is None:
        if output_dir:
            if filename is None:
                if "master_timestamp" in df.columns:
                    timestamp = df["master_timestamp"].iloc[0]
                    if isinstance(timestamp, str):
                        safe_time_str = timestamp.replace(":", "-").replace(" ", "_")
                    else:
                        safe_time_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    filename = f"dic_result_{safe_time_str}"
                else:
                    filename = "dic_result"

            save_path = output_path / f"{filename}.png"
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
            plt.close(current_fig)
        elif show:
            plt.show()
        else:
            plt.close(current_fig)
        return None
    else:
        return current_fig, current_ax, q
