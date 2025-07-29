from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image as PILImage


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
    cmap_name: str = "batlow",
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, object] | None:
    """Base function to plot DIC displacement vectors using numpy arrays.

    Args:
        x: X coordinates (seed points)
        y: Y coordinates (seed points)
        u: X displacement components
        v: Y displacement components
        magnitudes: Displacement magnitudes
        background_image: Optional background image array
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        scale: Quiver scale parameter
        scale_units: Units for quiver scaling
        width: Width of the quiver arrows
        headwidth: Headwidth for quiver arrows
        alpha: Alpha transparency for quiver arrows
        cmap_name: Name of colormap
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the figure
        ax: Optional matplotlib Axes to plot on
        fig: Optional matplotlib Figure to use
        title: Optional plot title

    Returns:
        Tuple of (figure, axes, quiver_object)

    Raises:
        ValueError: If input arrays have different lengths or are empty
    """
    # Input validation
    arrays = [x, y, u, v, magnitudes]
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("All input arrays must have the same length")

    if len(x) == 0:
        raise ValueError("Input arrays cannot be empty")

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
    max_magnitude = vmax if vmax is not None else np.max(magnitudes)
    norm = Normalize(vmin=vmin, vmax=max_magnitude)

    # Set up figure and axes
    if ax is not None:
        ax = ax
        fig = fig if fig is not None else ax.figure
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
        cmap=cmap,
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
    cmap_name: str = "batlow",
    s: float = 20,
    alpha: float = 0.8,
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, object]:
    """Plot DIC displacement data as a scatter plot colored by magnitude.

    Args:
        x: X coordinates (seed points)
        y: Y coordinates (seed points)
        magnitudes: Displacement magnitudes
        background_image: Optional background image array
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        cmap_name: Name of colormap
        s: Size of scatter points
        alpha: Alpha transparency
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the figure
        ax: Optional matplotlib Axes to plot on
        fig: Optional matplotlib Figure to use
        title: Optional plot title

    Returns:
        Tuple of (figure, axes, scatter_object)
    """
    # Input validation
    if len(x) != len(y) or len(x) != len(magnitudes):
        raise ValueError("All input arrays must have the same length")

    if len(x) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Handle colormap selection
    cmap = None
    if hasattr(cmc, cmap_name):
        cmap = getattr(cmc, cmap_name)
    elif cmap_name in plt.colormaps():
        cmap = plt.colormaps.get_cmap(cmap_name)
    else:
        cmap = plt.colormaps.get_cmap("viridis")

    # Set up figure and axes
    if ax is not None:
        ax = ax
        fig = fig if fig is not None else ax.figure
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
        cmap=cmap,
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
    else:
        ax.set_title("DIC Displacement Magnitude")

    # Disable axis grid and labels
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return fig, ax, scatter


def save_or_show_plot(
    fig: Figure,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    show: bool = False,
    dpi: int = 300,
    close_after_save: bool = True,
) -> None:
    """Save plot to file or show it interactively.

    Args:
        fig: Matplotlib figure to save/show
        output_dir: Directory to save plots
        filename: Filename (without extension)
        show: If True, show the plot interactively
        dpi: Dots per inch for saved figure
        close_after_save: If True, close figure after saving
    """
    if output_dir and filename:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"{filename}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        if close_after_save:
            plt.close(fig)
    elif show:
        plt.show()
    else:
        if close_after_save:
            plt.close(fig)


def visualize_dic_dataframe(
    df: pd.DataFrame,
    plot_type: str = "quiver",
    background_image: PILImage.Image | np.ndarray | None = None,
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
    **kwargs,
) -> tuple[Figure, Axes, object] | None:
    """Visualize DIC displacement data from a pandas DataFrame.

    Args:
        df: DataFrame containing DIC displacement data with original column names
        plot_type: Type of plot ("quiver" or "scatter")
        background_image: Optional background image
        output_dir: Directory to save plots
        filename: Custom filename for saved plot
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        scale: Quiver scale parameter (quiver plots only)
        scale_units: Units for quiver scaling (quiver plots only)
        width: Width of quiver arrows (quiver plots only)
        headwidth: Headwidth for quiver arrows (quiver plots only)
        alpha: Alpha transparency
        cmap_name: Name of colormap
        show: If True, show the plot interactively
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the figure
        ax: Optional matplotlib Axes to plot on
        fig: Optional matplotlib Figure to use
        **kwargs: Additional keyword arguments for plot functions

    Returns:
        If ax is provided, returns (fig, ax, plot_obj). Otherwise returns None.

    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Define column mapping
    columns_to_extract = {
        "seed_x_px": "x",
        "seed_y_px": "y",
        "displacement_x_px": "u",
        "displacement_y_px": "v",
        "displacement_magnitude_px": "V",
    }

    # Check for required columns
    required_columns = list(columns_to_extract.keys())
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Extract and rename columns
    plot_df = df[required_columns].rename(columns=columns_to_extract)

    # Convert to numpy arrays
    x = plot_df["x"].values
    y = plot_df["y"].values
    u = plot_df["u"].values
    v = plot_df["v"].values
    magnitudes = plot_df["V"].values

    # Handle background image
    bg_array = None
    if background_image is not None:
        if isinstance(background_image, PILImage.Image):
            bg_array = np.array(background_image)
        else:
            bg_array = background_image

    # Generate title from timestamp if available
    title = None
    if "master_timestamp" in df.columns and not df["master_timestamp"].isnull().all():
        timestamp = df["master_timestamp"].iloc[0]
        if isinstance(timestamp, str):
            title = f"DIC Displacement - {timestamp}"
        elif hasattr(timestamp, "strftime"):
            title = f"DIC Displacement - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    # Create plot based on type
    if plot_type.lower() == "quiver":
        result = plot_dic_vectors(
            x=x,
            y=y,
            u=u,
            v=v,
            magnitudes=magnitudes,
            background_image=bg_array,
            vmin=vmin,
            vmax=vmax,
            scale=scale,
            scale_units=scale_units,
            width=width,
            headwidth=headwidth,
            alpha=alpha,
            cmap_name=cmap_name,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
            fig=fig,
            title=title,
            **kwargs,
        )
    elif plot_type.lower() == "scatter":
        result = plot_dic_scatter(
            x,
            y,
            magnitudes,
            background_image=bg_array,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            alpha=alpha,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
            fig=fig,
            title=title,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'quiver' or 'scatter'")

    # Handle output if no axes were provided
    if ax is None:
        fig, ax, plot_obj = result

        # Generate filename if not provided
        if output_dir and not filename:
            if (
                "master_timestamp" in df.columns
                and not df["master_timestamp"].isnull().all()
            ):
                timestamp = df["master_timestamp"].iloc[0]
                if isinstance(timestamp, str):
                    safe_time_str = timestamp.replace(":", "-").replace(" ", "_")
                elif hasattr(timestamp, "strftime"):
                    safe_time_str = timestamp.strftime("%Y%m%d_%H%M%S")
                else:
                    safe_time_str = str(timestamp)
                filename = f"dic_{plot_type}_{safe_time_str}"
            else:
                filename = f"dic_{plot_type}"

        # Save or show plot
        save_or_show_plot(fig, output_dir, filename, show, dpi)
        return None
    else:
        return result


# Convenience functions for specific use cases
def plot_dic_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    magnitudes: np.ndarray,
    background_image: np.ndarray | None = None,
    plot_type: str = "quiver",
    **kwargs,
) -> tuple[Figure, Axes, object]:
    """Convenience function to plot DIC data from numpy arrays."""
    if plot_type.lower() == "quiver":
        return plot_dic_vectors(x, y, u, v, magnitudes, background_image, **kwargs)
    elif plot_type.lower() == "scatter":
        return plot_dic_scatter(x, y, magnitudes, background_image, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def plot_dic_from_dict(
    data_dict: dict,
    background_image: np.ndarray | None = None,
    plot_type: str = "quiver",
    **kwargs,
) -> tuple[Figure, Axes, object]:
    """Convenience function to plot DIC data from a dictionary with keys x, y, u, v, V."""
    required_keys = (
        ["x", "y", "u", "v", "V"] if plot_type == "quiver" else ["x", "y", "V"]
    )
    missing_keys = set(required_keys) - set(data_dict.keys())
    if missing_keys:
        raise ValueError(f"Dictionary missing required keys: {missing_keys}")

    if plot_type.lower() == "quiver":
        return plot_dic_vectors(
            data_dict["x"],
            data_dict["y"],
            data_dict["u"],
            data_dict["v"],
            data_dict["V"],
            background_image,
            **kwargs,
        )
    elif plot_type.lower() == "scatter":
        return plot_dic_scatter(
            data_dict["x"], data_dict["y"], data_dict["V"], background_image, **kwargs
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
