import io
import mimetypes
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from PIL import Image as PILImage

from .functions.visualization import (
    draw_quiver_on_image_cv2,
    plot_dic_scatter,
    plot_dic_vectors,
)
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


@require_http_methods(["GET"])
def visualize_dic(request, dic_id):
    """
    Visualize DIC displacement data as a plot.

    Extended query parameters (in addition to previous):
        - plot_type: 'quiver' or 'scatter' (default: 'quiver')
        - background: 'true' or 'false' (default: 'true')
        - cmap: colormap name (default: 'viridis')
        - vmin: minimum value for color normalization (float)
        - vmax: maximum value for color normalization (float)
        - scale: quiver scale (float or 'None' to let matplotlib decide)
        - scale_units: quiver scale units (default: 'xy')
        - width: quiver arrow width (float)
        - headwidth: quiver arrow headwidth (float)
        - quiver_alpha: alpha for quiver arrows (float 0-1)
        - image_alpha: alpha for background image (float 0-1)
        - figsize: comma separated width,height in inches (e.g. 10,8)
        - dpi: figure DPI (int)
        - filter_outliers: 'true'/'false' (apply percentile filtering)
        - tails_percentile: percentile for outlier trimming (0.01 = 1%)
        - min_velocity: minimum magnitude to keep (float)
        - subsample: integer, take every n-th vector to speed plotting/sampling
    """
    # Get DIC record
    dic = get_object_or_404(DIC, id=dic_id)

    # Check if DIC file exists
    if dic.result_file_path is None or not os.path.exists(dic.result_file_path):
        raise Http404("DIC HDF5 file not found")

    # Helper parsers
    def _parse_float(s, default=None):
        try:
            return float(s) if s is not None else default
        except (ValueError, TypeError):
            return default

    def _parse_int(s, default=None):
        try:
            return int(s) if s is not None else default
        except (ValueError, TypeError):
            return default

    def _parse_bool(s, default=False):
        if s is None:
            return default
        return str(s).lower() in ("1", "true", "yes", "y")

    # Parse query parameters
    plot_type = request.GET.get("plot_type", "quiver").lower()
    if plot_type not in ("quiver", "scatter"):
        return HttpResponse("Invalid plot_type. Use 'quiver' or 'scatter'.", status=400)

    show_background = _parse_bool(request.GET.get("background"), True)
    cmap_name = request.GET.get("cmap", "viridis")
    vmin = _parse_float(request.GET.get("vmin"), None)
    vmax = _parse_float(request.GET.get("vmax"), None)

    scale = request.GET.get("scale", None)
    if scale is not None and scale.lower() != "none":
        scale = _parse_float(scale, None)
    else:
        scale = None
    scale_units = request.GET.get("scale_units", "xy")
    width = _parse_float(request.GET.get("width"), 0.003)
    headwidth = _parse_float(request.GET.get("headwidth"), 2.5)
    quiver_alpha = _parse_float(request.GET.get("quiver_alpha"), 1.0)
    image_alpha = _parse_float(request.GET.get("image_alpha"), 0.7)

    figsize_param = request.GET.get("figsize", None)
    if figsize_param:
        try:
            w, h = [float(x) for x in figsize_param.split(",", 1)]
            figsize = (w, h)
        except Exception:
            figsize = (10, 8)
    else:
        figsize = (10, 8)

    dpi = _parse_int(request.GET.get("dpi"), 100)

    # Simple filtering params (performed locally, no external DB functions)
    filter_outliers = _parse_bool(request.GET.get("filter_outliers"), True)
    tails_percentile = _parse_float(request.GET.get("tails_percentile"), 0.01)
    min_velocity = _parse_float(request.GET.get("min_velocity"), -1.0)
    subsample = _parse_int(request.GET.get("subsample"), 1)
    if subsample is None or subsample < 1:
        subsample = 1

    # Read DIC data from HDF5
    try:
        with h5py.File(dic.result_file_path, "r") as f:
            points = f["points"][()] if "points" in f else None
            vectors = f["vectors"][()] if "vectors" in f else None
            magnitudes = f["magnitudes"][()] if "magnitudes" in f else None
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e

    if points is None or magnitudes is None:
        raise Http404("DIC HDF5 missing required datasets ('points' or 'magnitudes')")

    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    if vectors is not None:
        u = vectors[:, 0].astype(float)
        v = vectors[:, 1].astype(float)
    else:
        # create zero vectors if not present (scatter-only use-case)
        u = np.zeros_like(x)
        v = np.zeros_like(x)
    mag = magnitudes.astype(float)

    # Apply simple filtering locally
    mask = np.ones_like(mag, dtype=bool)
    if filter_outliers and 0.0 < tails_percentile < 0.5:
        lo, hi = np.percentile(
            mag, [100.0 * tails_percentile, 100.0 * (1.0 - tails_percentile)]
        )
        mask &= (mag >= lo) & (mag <= hi)
    if min_velocity is not None and min_velocity > 0:
        mask &= mag >= min_velocity

    # Apply mask
    x = x[mask]
    y = y[mask]
    u = u[mask]
    v = v[mask]
    mag = mag[mask]

    # Subsample spatially by taking every n-th vector
    if subsample > 1:
        idx = np.arange(0, len(x), subsample)
        x = x[idx]
        y = y[idx]
        u = u[idx]
        v = v[idx]
        mag = mag[idx]

    # Prepare background image if requested
    background_image = None
    if show_background:
        try:
            image_path = dic.master_image.file_path
            if os.path.exists(image_path):
                pil_image = PILImage.open(image_path)
                # rotate for tele cameras if necessary
                is_tele_camera = False
                if hasattr(dic.master_image, "camera") and dic.master_image.camera:
                    camera_name = dic.master_image.camera.camera_name
                    is_tele_camera = "PPCX_Tele" in camera_name or "Tele" in camera_name
                if is_tele_camera:
                    pil_image = pil_image.rotate(90, expand=True)
                background_image = np.array(pil_image)
        except Exception:
            background_image = None

    # Create figure with requested figsize/dpi and call plotting helpers
    plt.figure(figsize=figsize, dpi=dpi)

    if plot_type == "quiver":
        # plot_dic_vectors returns (fig, ax, q)
        fig, ax, _ = plot_dic_vectors(
            x,
            y,
            u,
            v,
            mag,
            background_image=background_image,
            vmin=vmin if vmin is not None else 0.0,
            vmax=vmax,
            scale=scale,
            scale_units=scale_units,
            width=width,
            headwidth=headwidth,
            quiver_alpha=quiver_alpha,
            image_alpha=image_alpha,
            cmap_name=cmap_name,
            figsize=figsize,
            dpi=dpi,
        )
    else:  # scatter
        fig, ax, _ = plot_dic_scatter(
            x,
            y,
            mag,
            background_image=background_image,
            vmin=vmin if vmin is not None else 0.0,
            vmax=vmax,
            cmap_name=cmap_name,
            figsize=figsize,
            dpi=dpi,
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

    return HttpResponse(buf.getvalue(), content_type="image/png")


@require_http_methods(["GET"])
def serve_dic_quiver(request, dic_id):
    """
    Generate and return a quiver image (PNG) drawn with OpenCV for a DIC record.

    Query parameters:
      - colormap: matplotlib colormap name (default: 'viridis')
      - arrow_length_scale: multiplier for vector display (float, default: 1.0)
      - arrow_thickness: integer thickness for arrows (default: 1)
      - alpha: arrow alpha (not used for cv2 arrows, kept for compatibility)
      - subsample: integer, take every n-th vector (default: 1)
      - background: true/false (default: true) - draw on background image if available
      - rotate_tele: true/false (default: true) - rotate background image for Tele cameras
    """
    dic = get_object_or_404(DIC, id=dic_id)

    if dic.result_file_path is None or not os.path.exists(dic.result_file_path):
        raise Http404("DIC HDF5 file not found")

    def _parse_float(s, default=None):
        try:
            return float(s) if s is not None else default
        except (ValueError, TypeError):
            return default

    def _parse_int(s, default=None):
        try:
            return int(s) if s is not None else default
        except (ValueError, TypeError):
            return default

    def _parse_bool(s, default=False):
        if s is None:
            return default
        return str(s).lower() in ("1", "true", "yes", "y")

    # params
    colormap = request.GET.get("colormap", "viridis")
    arrow_length_scale = _parse_float(request.GET.get("arrow_length_scale"), 1.0)
    arrow_thickness = _parse_int(request.GET.get("arrow_thickness"), 1)
    subsample = _parse_int(request.GET.get("subsample"), 1) or 1
    background = _parse_bool(request.GET.get("background"), True)
    rotate_tele = _parse_bool(request.GET.get("rotate_tele"), True)

    # read HDF5
    try:
        with h5py.File(dic.result_file_path, "r") as f:
            points = f["points"][()] if "points" in f else None
            vectors = f["vectors"][()] if "vectors" in f else None
            magnitudes = f["magnitudes"][()] if "magnitudes" in f else None
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e

    if points is None or magnitudes is None:
        raise Http404("DIC HDF5 missing required datasets ('points' or 'magnitudes')")

    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    if vectors is not None:
        u = vectors[:, 0].astype(float)
        v = vectors[:, 1].astype(float)
    else:
        u = np.zeros_like(x)
        v = np.zeros_like(x)
    mag = magnitudes.astype(float)

    # subsample
    if subsample > 1:
        idx = np.arange(0, len(x), subsample)
        x = x[idx]
        y = y[idx]
        u = u[idx]
        v = v[idx]
        mag = mag[idx]

    # Prepare background image if requested
    background_image = None
    if background:
        try:
            image_path = dic.master_image.file_path
            if image_path and os.path.exists(image_path):
                pil_image = PILImage.open(image_path)
                # rotate for tele cameras if necessary (only rotating image, not vectors)
                if (
                    rotate_tele
                    and hasattr(dic.master_image, "camera")
                    and dic.master_image.camera
                ):
                    camera_name = dic.master_image.camera.camera_name or ""
                    is_tele_camera = "PPCX_Tele" in camera_name or "Tele" in camera_name
                else:
                    is_tele_camera = False
                if is_tele_camera:
                    pil_image = pil_image.rotate(90, expand=True)
                # convert to numpy BGR as cv2 expects BGR; we'll convert later if needed
                background_image = np.array(pil_image)
                # convert RGB->BGR if needed (PIL gives RGB or L)
                if background_image.ndim == 3 and background_image.shape[2] == 3:
                    # convert RGB to BGR for cv2 processing
                    background_image = background_image[:, :, ::-1].copy()
                elif background_image.ndim == 2:
                    # grayscale -> convert to BGR
                    background_image = np.stack([background_image] * 3, axis=2)
        except Exception:
            background_image = None

    # If no background provided, create a white canvas sized from data extents
    if background_image is None:
        # compute bounds with some margin
        max_x = int(np.ceil(np.max(x))) if x.size else 200
        max_y = int(np.ceil(np.max(y))) if y.size else 200
        canvas_w = max(200, max_x + 10)
        canvas_h = max(200, max_y + 10)
        background_image = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # draw arrows on image using helper (returns BGR numpy image)
    try:
        out_bgr = draw_quiver_on_image_cv2(
            background_image,
            x,
            y,
            u,
            v,
            mag,
            colormap_name=colormap,
            arrow_length_scale=arrow_length_scale,
            arrow_thickness=arrow_thickness,
            alpha=1.0,
        )
    except Exception as e:
        raise Http404(f"Error drawing quiver with OpenCV: {e}") from e

    # encode PNG
    try:
        ok, buf = cv2.imencode(".png", out_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        png_bytes = buf.tobytes()
    except Exception:
        # fallback via PIL if cv2 not available for encode (shouldn't happen)
        try:
            out_rgb = out_bgr[:, :, ::-1]
            pil_out = PILImage.fromarray(out_rgb)
            bio = io.BytesIO()
            pil_out.save(bio, format="PNG")
            png_bytes = bio.getvalue()
        except Exception as e:
            raise Http404(f"Failed to encode PNG: {e}") from e

    response = HttpResponse(png_bytes, content_type="image/png")
    response["Content-Disposition"] = f'attachment; filename="dic_{dic_id}_quiver.png"'
    return response


@require_http_methods(["GET"])
def dic_visualizer(request, dic_id=None):
    """
    Render a simple DIC visualizer page.

    - Provides lightweight server-side filtering for DIC records (no heavy DB helpers).
    - Renders a list of matching DICs and a small form with plotting options.
    - When the user selects a DIC and options, the client loads the PNG from the
      existing `visualize_dic` endpoint (which returns a PNG image).
    """
    # Parse simple filter params from querystring
    q_reference_date = request.GET.get("reference_date")
    q_master_start = request.GET.get("master_timestamp_start")
    q_master_end = request.GET.get("master_timestamp_end")
    q_camera_id = request.GET.get("camera_id")
    q_camera_name = request.GET.get("camera_name")
    q_time_diff_min = request.GET.get("time_difference_min")
    q_time_diff_max = request.GET.get("time_difference_max")
    q_month = request.GET.get("month")

    # Build ORM queryset with safe/optional filters
    qs = DIC.objects.select_related("master_image__camera", "slave_image").all()

    if q_reference_date:
        try:
            dt = datetime.fromisoformat(q_reference_date)
            qs = qs.filter(reference_date=dt.date())
        except Exception:
            # ignore bad parse, leave filter out
            pass

    if q_master_start:
        try:
            dt = datetime.fromisoformat(q_master_start)
            qs = qs.filter(master_timestamp__gte=dt)
        except Exception:
            pass

    if q_master_end:
        try:
            dt = datetime.fromisoformat(q_master_end)
            qs = qs.filter(master_timestamp__lte=dt)
        except Exception:
            pass

    if q_camera_id:
        try:
            qs = qs.filter(master_image__camera__id=int(q_camera_id))
        except Exception:
            pass

    if q_camera_name:
        qs = qs.filter(master_image__camera__camera_name__icontains=q_camera_name)

    if q_time_diff_min:
        try:
            qs = qs.filter(time_difference_hours__gte=int(q_time_diff_min))
        except Exception:
            pass

    if q_time_diff_max:
        try:
            qs = qs.filter(time_difference_hours__lte=int(q_time_diff_max))
        except Exception:
            pass

    if q_month:
        try:
            m = int(q_month)
            if 1 <= m <= 12:
                qs = qs.filter(reference_date__month=m)
        except Exception:
            pass

    # Order and limit for a responsive page
    dic_list = qs.order_by("-master_timestamp")[:200]

    # Default plotting options to render in the form
    default_options = {
        "plot_type": request.GET.get("plot_type", "quiver"),
        "background": request.GET.get("background", "true"),
        "cmap": request.GET.get("cmap", "viridis"),
        "vmin": request.GET.get("vmin", ""),
        "vmax": request.GET.get("vmax", ""),
        "scale": request.GET.get("scale", ""),
        "scale_units": request.GET.get("scale_units", "xy"),
        "width": request.GET.get("width", "0.003"),
        "headwidth": request.GET.get("headwidth", "2.5"),
        "quiver_alpha": request.GET.get("quiver_alpha", "1.0"),
        "image_alpha": request.GET.get("image_alpha", "0.7"),
        "figsize": request.GET.get("figsize", "10,8"),
        "dpi": request.GET.get("dpi", "150"),
        "filter_outliers": request.GET.get("filter_outliers", "true"),
        "tails_percentile": request.GET.get("tails_percentile", "0.01"),
        "min_velocity": request.GET.get("min_velocity", ""),
        "subsample": request.GET.get("subsample", "1"),
    }

    context = {
        "dic_list": dic_list,
        "selected_dic_id": dic_id,
        "plot_opts": default_options,
        # helper to build visualize URL in template
        "visualize_base_url": reverse("visualize_dic", kwargs={"dic_id": 0}).rstrip(
            "0"
        ),
    }
    return render(request, "ppcx_app/dic_visualizer.html", context)
