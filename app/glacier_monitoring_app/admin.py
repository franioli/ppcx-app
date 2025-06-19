from django.contrib import admin
from django.contrib.gis import admin as gis_admin
from django.contrib import admin
from django.utils.html import format_html
from .models import Image
from django.urls import reverse

from .models import Camera, CameraCalibration, DICAnalysis, DICResult, Image


@admin.register(Camera)
class CameraAdmin(gis_admin.GISModelAdmin):
    list_display = ("camera_name", "model", "installation_date")
    search_fields = ("camera_name", "serial_number")
    list_filter = ("installation_date",)


@admin.register(CameraCalibration)
class CameraCalibrationAdmin(admin.ModelAdmin):
    list_display = (
        "camera",
        "calibration_date",
        "colmap_model_name",
        "is_active",
    )
    list_filter = (
        "camera",
        "is_active",
        "calibration_date",
        "colmap_model_id",
    )
    search_fields = (
        "camera__camera_name",
        "notes",
    )

    def save_model(self, request, obj, form, change):
        # Ensure only one active calibration per camera
        if obj.is_active:
            CameraCalibration.objects.filter(camera=obj.camera, is_active=True).exclude(
                pk=obj.pk
            ).update(is_active=False)
        super().save_model(request, obj, form, change)


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ("id", "camera", "acquisition_timestamp", "file_path", "view_image")
    list_filter = ["camera", "acquisition_timestamp"]
    search_fields = ["camera__camera_name", "file_path"]
    date_hierarchy = "acquisition_timestamp"

    def view_image(self, obj):
        if obj.file_path and obj.id:
            url = reverse('serve_image', args=[obj.id])
            return format_html(
                '<a href="{}" target="_blank">View Image</a>', 
                url
            )
        return ""
    view_image.short_description = "Preview"


class DICResultInline(admin.TabularInline):
    model = DICResult
    extra = 0


@admin.register(DICAnalysis)
class DICAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        "analysis_timestamp",
        "master_timestamp",
        "slave_timestamp",
    ]
    list_filter = [
        "analysis_timestamp",
    ]
    search_fields = [
        "master_image_path",
        "slave_image_path",
    ]
    inlines = [DICResultInline]


@admin.register(DICResult)
class DICResultAdmin(admin.ModelAdmin):
    list_display = []
    list_filter = [
        "analysis__time_difference_hours",
    ]
    search_fields = [
        "analysis_timestamp",
        "analysis__master_image_path",
        "analysis__master_timestamp",
        "analysis__slave_image_path",
        "analysis__slave_timestamp",
        "analysis__time_difference_hours",
    ]
