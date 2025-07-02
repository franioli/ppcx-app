"""
URL configuration for planpincieux project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path
from ppcx_app import views

urlpatterns = [
    path("", views.home, name="home"),  # This handles the root URL
    path("admin/", admin.site.urls),
    path("API/", include("ppcx_app.urls")),  # All API endpoints under /API/
    path("dic_visualizer/", views.dic_visualizer, name="dic_visualizer"),
    path(
        "dic_visualizer/<int:dic_id>/",
        views.dic_visualizer,
        name="dic_visualizer_with_id",
    ),
]
