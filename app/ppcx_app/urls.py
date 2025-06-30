from django.urls import path

from . import views

urlpatterns = [
    path("images/<int:image_id>/", views.serve_image, name="serve_image"),
    path("dic/<int:dic_id>/", views.serve_dic_h5, name="serve_dic_h5"),
]
