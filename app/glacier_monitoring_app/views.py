import mimetypes
import os

from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from .models import Image


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
