import mimetypes
import os

import h5py
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from .models import DIC, Image


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
