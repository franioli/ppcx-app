from django.urls import path

from . import views

urlpatterns = [
    path("image/<int:image_id>/", views.serve_image, name="serve_image"),
]
