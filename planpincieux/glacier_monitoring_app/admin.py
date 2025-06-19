from django.contrib import admin
from django.contrib.gis import admin as gis_admin

from .models import Camera, CameraCalibration, DICAnalysis, DICResult, Image


@admin.register(Camera)
class CameraAdmin(gis_admin.GISModelAdmin):
    list_display = ("camera_name", "model", "installation_date")
    search_fields = ("camera_name", "serial_number")
    list_filter = ("installation_date",)


@admin.register(CameraCalibration)
class CameraCalibrationAdmin(admin.ModelAdmin):
    list_display = ("camera", "calibration_date", "colmap_model_name", "is_active")
    list_filter = ("camera", "is_active", "calibration_date", "colmap_model_id")
    search_fields = ("camera__camera_name", "notes")

    def save_model(self, request, obj, form, change):
        # Ensure only one active calibration per camera
        if obj.is_active:
            CameraCalibration.objects.filter(camera=obj.camera, is_active=True).exclude(
                pk=obj.pk
            ).update(is_active=False)
        super().save_model(request, obj, form, change)


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ("id", "camera", "acquisition_timestamp", "file_path")
    list_filter = ("camera", "acquisition_timestamp")
    search_fields = ("file_path", "camera__camera_name")
    date_hierarchy = "acquisition_timestamp"


class DICResultInline(admin.TabularInline):
    model = DICResult
    extra = 0
    fields = (
        "seed_x_ref_px",
        "seed_y_ref_px",
        "displacement_x_px",
        "displacement_y_px",
        "correlation_score",
        "status_flag",
    )
    readonly_fields = ("displacement_x_px", "displacement_y_px")


@admin.register(DICAnalysis)
class DICAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "reference_image", "secondary_image", "time_difference_hours")
    list_filter = ("software_used", "analysis_timestamp")
    search_fields = ("reference_image__file_path", "secondary_image__file_path")
    inlines = [DICResultInline]


@admin.register(DICResult)
class DICResultAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "analysis",
        "seed_x_ref_px",
        "seed_y_ref_px",
        "displacement_x_px",
        "displacement_y_px",
        "correlation_score",
    )
    list_filter = ("status_flag", "analysis")
    search_fields = ("analysis__id",)
