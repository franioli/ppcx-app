import io
import mimetypes
import os
from datetime import datetime
from typing import Any

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

# optional presence of cv2 for encoding / fallbacks
try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    HAS_CV2 = False


def _parse_float(s: str | None, default: float | None = None) -> float | None:
    try:
        return float(s) if s is not None and s != "" else default
    except (ValueError, TypeError):
        return default


def _parse_int(s: str | None, default: int | None = None) -> int | None:
    try:
        return int(s) if s is not None and s != "" else default
    except (ValueError, TypeError):
        return default


def _parse_bool(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).lower() in ("1", "true", "yes", "y")


@require_http_methods(["GET"])
def home(request) -> HttpResponse:
    return HttpResponse("Welcome to the Planpincieux API. Use /API/ for endpoints.")


@require_http_methods(["GET"])
def serve_image(request, image_id: int) -> HttpResponse:
    """Serve image files by Image.id (inline)."""
    image = get_object_or_404(Image, id=image_id)
    if not os.path.exists(image.file_path):
        raise Http404("Image file not found")
    try:
        with open(image.file_path, "rb") as f:
            content = f.read()
        content_type, _ = mimetypes.guess_type(image.file_path)
        content_type = content_type or "application/octet-stream"
        response = HttpResponse(content, content_type=content_type)
        response["Content-Disposition"] = (
            f'inline; filename="{os.path.basename(image.file_path)}"'
        )
        return response
    except OSError:
        raise Http404("Could not read image file")


@require_http_methods(["GET"])
def serve_dic_h5(request, dic_id: int) -> HttpResponse:
    """
    Return DIC HDF5 content as JSON for the specified DIC id.
    Query params: none required.
    """
    dic = get_object_or_404(DIC, id=dic_id)
    h5_path = dic.result_file_path
    if not h5_path or not os.path.exists(h5_path):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5_path, "r") as f:
            data: dict[str, Any] = {
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
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e


@require_http_methods(["GET"])
def serve_dic_h5_as_csv(request, dic_id: int) -> HttpResponse:
    """Return DIC data as CSV (x,y,u,v,magnitude)."""
    dic = get_object_or_404(DIC, id=dic_id)
    h5_path = dic.result_file_path
    if not h5_path or not os.path.exists(h5_path):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5_path, "r") as f:
            points = f["points"][()] if "points" in f else None
            vectors = f["vectors"][()] if "vectors" in f else None
            magnitudes = f["magnitudes"][()] if "magnitudes" in f else None

        if points is None or magnitudes is None:
            return HttpResponse("No DIC data available", status=404)

        csv_lines = ["x,y,u,v,magnitude"]
        for i in range(len(points)):
            x, y = points[i]
            u, v = vectors[i] if vectors is not None else (0.0, 0.0)
            magnitude = magnitudes[i]
            csv_lines.append(f"{x},{y},{u},{v},{magnitude}")

        response = HttpResponse("\n".join(csv_lines), content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="dic_{dic_id}.csv"'
        return response
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e


def load_and_filter_dic(
    dic: DIC,
    *,
    filter_outliers: bool = True,
    tails_percentile: float = 0.01,
    min_velocity: float | None = None,
    subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read DIC HDF5 for `dic` and apply filtering + subsampling.

    Returns: x, y, u, v, mag (all numpy arrays float).
    Raises Http404 on errors or missing required datasets.
    """
    h5 = dic.result_file_path
    if not h5 or not os.path.exists(h5):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5, "r") as f:
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

    # Build mask and apply filters
    mask = np.ones_like(mag, dtype=bool)
    if filter_outliers and 0.0 < tails_percentile < 0.5:
        lo, hi = np.percentile(
            mag, [100.0 * tails_percentile, 100.0 * (1.0 - tails_percentile)]
        )
        mask &= (mag >= lo) & (mag <= hi)
    if min_velocity is not None and min_velocity > 0:
        mask &= mag >= min_velocity

    x = x[mask]
    y = y[mask]
    u = u[mask]
    v = v[mask]
    mag = mag[mask]

    if subsample is None or subsample < 1:
        subsample = 1
    if subsample > 1:
        idx = np.arange(0, len(x), subsample)
        x = x[idx]
        y = y[idx]
        u = u[idx]
        v = v[idx]
        mag = mag[idx]

    return x, y, u, v, mag


@require_http_methods(["GET"])
def visualize_dic(request, dic_id: int) -> HttpResponse:
    """
    Generate PNG visualization (matplotlib) for a DIC record.

    Query parameters (all parsed at start):
      - plot_type: 'quiver'|'scatter' (default 'quiver')
      - background: true/false (default true)
      - cmap: colormap (default 'viridis')
      - vmin / vmax: min/max magnitude for colormap (float)
      - min_velocity: minimum magnitude to keep (float)
      - filter_outliers: true/false (default true)
      - tails_percentile: percentile for trimming (default 0.01)
      - subsample: int (default 1)
      - scale / scale_units / width / headwidth / quiver_alpha / image_alpha
      - figsize: "W,H" and dpi
    """
    # parse params
    plot_type = request.GET.get("plot_type", "quiver").lower()
    if plot_type not in ("quiver", "scatter"):
        return HttpResponse("Invalid plot_type. Use 'quiver' or 'scatter'.", status=400)

    show_background = _parse_bool(request.GET.get("background"), True)
    cmap_name = request.GET.get("cmap", "viridis")
    vmin = _parse_float(request.GET.get("vmin"), None)
    vmax = _parse_float(request.GET.get("vmax"), None)

    scale_raw = request.GET.get("scale", None)
    scale = (
        None
        if scale_raw is None or scale_raw.lower() == "none"
        else _parse_float(scale_raw, None)
    )
    scale_units = request.GET.get("scale_units", "xy")
    width = _parse_float(request.GET.get("width"), 0.003) or 0.003
    headwidth = _parse_float(request.GET.get("headwidth"), 2.5) or 2.5
    quiver_alpha = _parse_float(request.GET.get("quiver_alpha"), 1.0) or 1.0
    image_alpha = _parse_float(request.GET.get("image_alpha"), 0.7) or 0.7

    # filtering params
    min_velocity = _parse_float(request.GET.get("min_velocity"), None)
    filter_outliers = _parse_bool(request.GET.get("filter_outliers"), True)
    tails_percentile = _parse_float(request.GET.get("tails_percentile"), 0.01) or 0.01
    subsample = _parse_int(request.GET.get("subsample"), 1) or 1

    figsize_param = request.GET.get("figsize", "")
    if figsize_param:
        try:
            w, h = [float(x) for x in figsize_param.split(",", 1)]
            figsize = (w, h)
        except Exception:
            figsize = (10.0, 8.0)
    else:
        figsize = (10.0, 8.0)
    dpi = _parse_int(request.GET.get("dpi"), 150) or 150

    dic = get_object_or_404(DIC, id=dic_id)

    # load and filter data
    x, y, u, v, mag = load_and_filter_dic(
        dic,
        filter_outliers=filter_outliers,
        tails_percentile=tails_percentile,
        min_velocity=min_velocity
        if (min_velocity is not None and min_velocity > 0)
        else None,
        subsample=subsample,
    )

    # background image (may be rotated for Tele cameras) - only rotate image, not vectors
    background_image: np.ndarray | None = None
    if show_background:
        try:
            image_path = dic.master_image.file_path if dic.master_image else None
            if image_path and os.path.exists(image_path):
                pil_image = PILImage.open(image_path)
                is_tele_camera = False
                if hasattr(dic.master_image, "camera") and dic.master_image.camera:
                    cam_name = dic.master_image.camera.camera_name or ""
                    is_tele_camera = "PPCX_Tele" in cam_name or "Tele" in cam_name
                if is_tele_camera:
                    pil_image = pil_image.rotate(90, expand=True)
                background_image = np.array(pil_image)
        except Exception:
            background_image = None

    # plotting
    plt.figure(figsize=figsize, dpi=dpi)
    if plot_type == "quiver":
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
    else:
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

    # title
    if dic.master_timestamp and dic.slave_timestamp:
        master_date = dic.master_timestamp.strftime("%Y-%m-%d %H:%M")
        slave_date = dic.slave_timestamp.strftime("%Y-%m-%d %H:%M")
        title = f"DIC #{dic_id}: {master_date} → {slave_date}"
        if dic.dt_hours:
            title += f" ({dic.dt_hours} hours)"
        ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="jpg",
        bbox_inches="tight",
    )
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/jpeg")


@require_http_methods(["GET"])
def serve_dic_quiver(request, dic_id: int) -> HttpResponse:
    """
    Generate and return a quiver PNG drawn with OpenCV (or PIL fallback).

    Query parameters parsed at start:
      - colormap (str)
      - arrow_length_scale (float) or omitted for auto
      - arrow_thickness (int) or omitted for auto
      - subsample (int)
      - rotate_tele (bool) default True (rotate background image only)
      - filter_outliers (bool), tails_percentile (float), min_velocity (float)
      - vmin / vmax (float) clip for color mapping
    """
    # parse params
    colormap = request.GET.get("colormap", "viridis")
    arrow_length_scale = _parse_float(request.GET.get("arrow_length_scale"), None)
    arrow_thickness = _parse_int(request.GET.get("arrow_thickness"), None)
    subsample = _parse_int(request.GET.get("subsample"), 1) or 1
    rotate_tele = _parse_bool(request.GET.get("rotate_tele"), True)

    vmin = _parse_float(request.GET.get("vmin"), None)
    vmax = _parse_float(request.GET.get("vmax"), None)
    min_velocity = _parse_float(request.GET.get("min_velocity"), None)
    filter_outliers = _parse_bool(request.GET.get("filter_outliers"), True)
    tails_percentile = _parse_float(request.GET.get("tails_percentile"), 0.01) or 0.01

    dic = get_object_or_404(DIC, id=dic_id)

    x, y, u, v, mag = load_and_filter_dic(
        dic,
        filter_outliers=filter_outliers,
        tails_percentile=tails_percentile,
        min_velocity=min_velocity
        if (min_velocity is not None and min_velocity > 0)
        else None,
        subsample=subsample,
    )

    # apply clip for color mapping if requested (does not change geometry)
    if (vmin is not None) or (vmax is not None):
        clip_lo = vmin if vmin is not None else (mag.min() if mag.size else 0.0)
        clip_hi = vmax if vmax is not None else (mag.max() if mag.size else clip_lo)
        if clip_hi < clip_lo:
            clip_lo, clip_hi = clip_hi, clip_lo
        mag = np.clip(mag, clip_lo, clip_hi)

    # background image (rotated for Tele cameras only)
    background_image = None
    try:
        image_path = dic.master_image.file_path if dic.master_image else None
        if image_path and os.path.exists(image_path):
            pil_image = PILImage.open(image_path)
            is_tele_camera = False
            if hasattr(dic.master_image, "camera") and dic.master_image.camera:
                cam_name = dic.master_image.camera.camera_name or ""
                is_tele_camera = "PPCX_Tele" in cam_name or "Tele" in cam_name
            if is_tele_camera and rotate_tele:
                pil_image = pil_image.rotate(90, expand=True)
            background_image = np.array(pil_image)
            # ensure BGR uint8 for OpenCV helper
            if background_image.ndim == 3 and background_image.shape[2] == 3:
                background_image = background_image[:, :, ::-1].copy()
            elif background_image.ndim == 2:
                background_image = np.stack([background_image] * 3, axis=2)
    except Exception:
        background_image = None

    # if no background create white canvas based on extents
    if background_image is None:
        max_x = int(np.ceil(np.max(x))) if x.size else 200
        max_y = int(np.ceil(np.max(y))) if y.size else 200
        canvas_w = max(200, max_x + 10)
        canvas_h = max(200, max_y + 10)
        background_image = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # draw using helper
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
        raise Http404(f"Error drawing quiver: {e}") from e

    # encode PNG (prefer cv2 if available)
    try:
        if HAS_CV2 and cv2 is not None:
            ok, buf = cv2.imencode(".png", out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            png_bytes = buf.tobytes()
        else:
            out_rgb = out_bgr[:, :, ::-1]
            pil_out = PILImage.fromarray(out_rgb)
            bio = io.BytesIO()
            pil_out.save(bio, format="PNG", optimize=True, compress_level=3)
            png_bytes = bio.getvalue()
    except Exception as e:
        raise Http404(f"Failed to encode PNG: {e}") from e

    response = HttpResponse(png_bytes, content_type="image/png")
    response["Content-Disposition"] = f'attachment; filename="dic_{dic_id}_quiver.png"'
    return response


@require_http_methods(["GET"])
def dic_visualizer(request, dic_id: int | None = None) -> HttpResponse:
    """
    Render a small DIC visualizer page with selectable DIC list and default plot options.
    Querystring can be used to filter list and set defaults. See template for usage.
    """
    # simple filters parsed at start
    q_reference_date = request.GET.get("reference_date")
    q_master_start = request.GET.get("master_timestamp_start")
    q_master_end = request.GET.get("master_timestamp_end")
    q_camera_id = request.GET.get("camera_id")
    q_camera_name = request.GET.get("camera_name")
    q_time_diff_min = request.GET.get("time_difference_min")
    q_time_diff_max = request.GET.get("time_difference_max")
    q_month = request.GET.get("month")

    qs = DIC.objects.select_related("master_image__camera", "slave_image").all()

    if q_reference_date:
        try:
            dt = datetime.fromisoformat(q_reference_date)
            qs = qs.filter(reference_date=dt.date())
        except Exception:
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
            qs = qs.filter(dt_hours__gte=int(q_time_diff_min))
        except Exception:
            pass

    if q_time_diff_max:
        try:
            qs = qs.filter(dt_hours__lte=int(q_time_diff_max))
        except Exception:
            pass

    if q_month:
        try:
            m = int(q_month)
            if 1 <= m <= 12:
                qs = qs.filter(reference_date__month=m)
        except Exception:
            pass

    dic_list = qs.order_by("-master_timestamp")[:200]

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

    # build visualize base pattern with placeholder for dic id
    base = reverse("visualize_dic", kwargs={"dic_id": 0})
    visualize_base_url = base.replace("/0/", "/{dic_id}/")

    context = {
        "dic_list": dic_list,
        "selected_dic_id": dic_id,
        "plot_opts": default_options,
        "visualize_base_url": visualize_base_url,
    }
    return render(request, "ppcx_app/dic_visualizer.html", context)
