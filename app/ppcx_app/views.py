import io
import mimetypes
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_http_methods
from PIL import Image as PILImage

from .functions.visualization import plot_dic_scatter, plot_dic_vectors
from .models import DIC, Image

matplotlib.use("Agg")  # Use non-interactive backend


def home(request):
    return HttpResponse("Welcome to the Planpincieux API. Use /API/ for endpoints.")


@require_http_methods(["GET"])
def serve_image(request, image_id):
    """Serve image files by image ID."""
    image = get_object_or_404(Image, id=image_id)

    if not os.path.exists(image.file_path):
        raise Http404("Image file not found")

    try:
        with open(image.file_path, "rb") as f:
            content = f.read()

        content_type, _ = mimetypes.guess_type(image.file_path)
        if not content_type:
            content_type = "application/octet-stream"

        response = HttpResponse(content, content_type=content_type)
        response["Content-Disposition"] = (
            f'inline; filename="{os.path.basename(image.file_path)}"'
        )
        return response

    except OSError:
        raise Http404("Could not read image file")


@require_http_methods(["GET"])
def serve_dic_h5(request, dic_id):
    """
    Serve DIC HDF5 data as JSON by DIC ID.
    Returns the contents of the HDF5 file as a JSON response.
    """
    dic = get_object_or_404(DIC, id=dic_id)
    h5_path = dic.result_file_path

    if not os.path.exists(h5_path):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5_path, "r") as f:
            data = {
                "points": f["points"][()].tolist() if "points" in f else None,
                "vectors": f["vectors"][()].tolist() if "vectors" in f else None,
                "magnitudes": f["magnitudes"][()].tolist()
                if "magnitudes" in f
                else None,
                "max_magnitude": float(f["max_magnitude"][()])
                if "max_magnitude" in f
                else None,
            }
        return JsonResponse(data)
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}")


@require_http_methods(["GET"])
def serve_dic_h5_as_csv(request, dic_id):
    """Serve DIC data as CSV by DIC ID."""
    dic = get_object_or_404(DIC, id=dic_id)
    h5_path = dic.result_file_path

    if not os.path.exists(h5_path):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5_path, "r") as f:
            points = f["points"][()] if "points" in f else None
            vectors = f["vectors"][()] if "vectors" in f else None
            magnitudes = f["magnitudes"][()] if "magnitudes" in f else None

        if points is None or vectors is None or magnitudes is None:
            return HttpResponse("No DIC data available", status=404)

        # Prepare CSV content
        csv_lines = ["x,y,u,v,magnitude"]
        for i in range(len(points)):
            x, y = points[i]
            u, v = vectors[i]
            magnitude = magnitudes[i]
            csv_lines.append(f"{x},{y},{u},{v},{magnitude}")

        response = HttpResponse("\n".join(csv_lines), content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="dic_{dic_id}.csv"'
        return response

    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}")


# Add this new function to your views.py
@require_http_methods(["GET"])
def dic_visualizer(request, dic_id=None):
    """Serve the DIC visualization tool."""
    context = {}
    if dic_id:
        context["dic_id"] = dic_id
    return render(request, "ppcx_app/dic_visualizer.html", context)


@require_http_methods(["GET"])
def visualize_dic(request, dic_id):
    """
    Visualize DIC displacement data as a plot.
    Returns a PNG image of the plot.

    Query parameters:
        - plot_type: 'quiver' or 'scatter' (default: 'quiver')
        - background: 'true' or 'false' to show background image (default: 'true')
        - cmap: colormap name (default: 'viridis')
        - vmax: maximum value for colorbar (default: auto)
        - filter_outliers: 'true' or 'false' (default: 'false')
    """
    # Get DIC record
    dic = get_object_or_404(DIC, id=dic_id)

    # Check if DIC file exists
    if dic.result_file_path is None or not os.path.exists(dic.result_file_path):
        raise Http404("DIC HDF5 file not found")

    # Parse query parameters
    plot_type = request.GET.get("plot_type", "quiver").lower()
    if plot_type not in ["quiver", "scatter"]:
        return HttpResponse("Invalid plot_type. Use 'quiver' or 'scatter'.", status=400)

    show_background = request.GET.get("background", "true").lower() == "true"
    cmap_name = request.GET.get("cmap", "viridis")

    try:
        vmax = float(request.GET.get("vmax", 0)) if request.GET.get("vmax") else None
    except ValueError:
        vmax = None

    filter_outliers = request.GET.get("filter_outliers", "false").lower() == "true"

    # Read DIC data
    try:
        with h5py.File(dic.result_file_path, "r") as f:
            points = f["points"][()]
            x = points[:, 0]
            y = points[:, 1]

            if plot_type == "quiver":
                vectors = f["vectors"][()]
                u = vectors[:, 0]
                v = vectors[:, 1]

            magnitudes = f["magnitudes"][()]
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e

    # Get background image if requested
    background_image = None

    if show_background:
        try:
            # Get master image
            image_path = dic.master_image.file_path

            # Check if camera is Tele (portrait mode)
            # TODO: In future, this should be read from camera orientation field or image EXIF
            is_tele_camera = False
            if hasattr(dic.master_image, "camera") and dic.master_image.camera:
                camera_name = dic.master_image.camera.camera_name
                is_tele_camera = "PPCX_Tele" in camera_name or "Tele" in camera_name

            if os.path.exists(image_path):
                pil_image = PILImage.open(image_path)

                # For Tele cameras, rotate the image (not the data)
                if is_tele_camera:
                    pil_image = pil_image.rotate(90, expand=True)

                background_image = np.array(pil_image)
        except Exception:
            # Continue without background if there's an error
            pass

    # Create figure and plot
    plt.figure(figsize=(10, 8), dpi=100)

    if plot_type == "quiver":
        fig, ax, _ = plot_dic_vectors(
            x,
            y,
            u,
            v,
            magnitudes,
            background_image=background_image,
            vmax=vmax,
            cmap_name=cmap_name,
        )
    else:  # scatter
        fig, ax, _ = plot_dic_scatter(
            x,
            y,
            magnitudes,
            background_image=background_image,
            vmax=vmax,
            cmap_name=cmap_name,
        )

    # Add title with DIC info
    if dic.master_timestamp and dic.slave_timestamp:
        master_date = dic.master_timestamp.strftime("%Y-%m-%d %H:%M")
        slave_date = dic.slave_timestamp.strftime("%Y-%m-%d %H:%M")
        title = f"DIC #{dic_id}: {master_date} → {slave_date}"
        if dic.time_difference_hours:
            title += f" ({dic.time_difference_hours} hours)"
        ax.set_title(title)

    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # Return image
    return HttpResponse(buf.getvalue(), content_type="image/png")
