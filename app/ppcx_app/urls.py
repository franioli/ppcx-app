from django.urls import path

from . import views

urlpatterns = [
    path("images/<int:image_id>/", views.serve_image, name="serve_image"),
    path("dic/<int:dic_id>/", views.serve_dic_h5, name="serve_dic_h5"),
    path(
        "dic/<int:dic_id>/csv/", views.serve_dic_h5_as_csv, name="serve_dic_h5_as_csv"
    ),
    path("dic/<int:dic_id>/plot/", views.visualize_dic, name="visualize_dic"),
]
