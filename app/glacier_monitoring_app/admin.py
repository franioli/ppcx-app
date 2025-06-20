import json

from django import forms
from django.contrib import admin
from django.contrib.gis import admin as gis_admin
from django.contrib.gis.forms import OSMWidget
from django.urls import reverse
from django.utils.html import format_html

from .models import Camera, CameraCalibration, DICAnalysis, DICResult, Image

# ========== Cameras and Calibrations ==========


class CameraAdminForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = "__all__"
        widgets = {"location": OSMWidget(attrs={"map_width": 800, "map_height": 500})}


@admin.register(Camera)
class CameraAdmin(gis_admin.GISModelAdmin):
    form = CameraAdminForm
    list_display = (
        "camera_name",
        "model",
        "lens",
        "focal_length_mm",
        "installation_date",
        "image_count",
    )
    search_fields = ("camera_name", "serial_number")
    list_filter = ("installation_date",)

    def image_count(self, obj):
        """Display the number of images for this camera"""
        return obj.images.count()

    image_count.short_description = "Number of Images"

    def get_queryset(self, request):
        """Optimize queries by prefetching related images"""
        queryset = super().get_queryset(request)
        return queryset.prefetch_related("images")


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


# ========== Images ==========


# Custom filters for ImageAdmin
class TimeOfDayFilter(admin.SimpleListFilter):
    title = "time of day"
    parameter_name = "time_of_day"

    def lookups(self, request, model_admin):
        return (
            ("0", "00:00 - 01:00"),
            ("1", "01:00 - 02:00"),
            ("2", "02:00 - 03:00"),
            ("3", "03:00 - 04:00"),
            ("4", "04:00 - 05:00"),
            ("5", "05:00 - 06:00"),
            ("6", "06:00 - 07:00"),
            ("7", "07:00 - 08:00"),
            ("8", "08:00 - 09:00"),
            ("9", "09:00 - 10:00"),
            ("10", "10:00 - 11:00"),
            ("11", "11:00 - 12:00"),
            ("12", "12:00 - 13:00"),
            ("13", "13:00 - 14:00"),
            ("14", "14:00 - 15:00"),
            ("15", "15:00 - 16:00"),
            ("16", "16:00 - 17:00"),
            ("17", "17:00 - 18:00"),
            ("18", "18:00 - 19:00"),
            ("19", "19:00 - 20:00"),
            ("20", "20:00 - 21:00"),
            ("21", "21:00 - 22:00"),
            ("22", "22:00 - 23:00"),
            ("23", "23:00 - 24:00"),
        )

    def queryset(self, request, queryset):
        if self.value():
            hour = int(self.value())
            return queryset.filter(acquisition_timestamp__hour=hour)


class MonthFilter(admin.SimpleListFilter):
    title = "month"
    parameter_name = "month"

    def lookups(self, request, model_admin):
        return (
            ("1", "January"),
            ("2", "February"),
            ("3", "March"),
            ("4", "April"),
            ("5", "May"),
            ("6", "June"),
            ("7", "July"),
            ("8", "August"),
            ("9", "September"),
            ("10", "October"),
            ("11", "November"),
            ("12", "December"),
        )

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(acquisition_timestamp__month=self.value())


class YearFilter(admin.SimpleListFilter):
    title = "year"
    parameter_name = "year"

    def lookups(self, request, model_admin):
        # Get all unique years from the acquisition_timestamp field
        years = Image.objects.dates("acquisition_timestamp", "year").values_list(
            "acquisition_timestamp__year", flat=True
        )
        return [(str(year), str(year)) for year in sorted(years, reverse=True)]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(acquisition_timestamp__year=self.value())


class ImageAdminForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = "__all__"
        widgets = {
            "exif_data": forms.Textarea(
                attrs={
                    "rows": 20,
                    "cols": 80,
                    "readonly": True,
                    "style": "font-family: monospace; font-size: 12px;",
                }
            )
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.exif_data:
            try:
                # Pretty format the JSON data
                formatted_json = json.dumps(
                    self.instance.exif_data,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                self.fields["exif_data"].initial = formatted_json
            except (TypeError, ValueError):
                # If it's not valid JSON, leave as is
                pass


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ("id", "camera", "acquisition_timestamp", "file_name", "view_image")
    list_filter = [
        "camera",
        "acquisition_timestamp",
        YearFilter,
        MonthFilter,
        TimeOfDayFilter,
    ]
    search_fields = ["camera__camera_name", "file_name"]
    date_hierarchy = "acquisition_timestamp"
    readonly_fields = ("formatted_exif_data",)

    def formatted_exif_data(self, obj):
        """Display formatted EXIF data in a readable way"""
        if obj.exif_data:
            try:
                formatted_json = json.dumps(
                    obj.exif_data, indent=2, ensure_ascii=False, sort_keys=True
                )
                return format_html(
                    '<pre style="background: #f8f8f8; padding: 10px; border: 1px solid #ddd; '
                    "border-radius: 4px; font-family: monospace; font-size: 12px; "
                    'max-height: 400px; overflow-y: auto;">{}</pre>',
                    formatted_json,
                )
            except (TypeError, ValueError):
                return format_html(
                    '<pre style="background: #fff2f2; padding: 10px; border: 1px solid #fdd; '
                    'border-radius: 4px; color: #d00;">Invalid JSON data</pre>'
                )
        return "No EXIF data"

    formatted_exif_data.short_description = "EXIF Data (Formatted)"

    def view_image(self, obj):
        if obj.file_path and obj.id:
            url = reverse("serve_image", args=[obj.id])
            return format_html('<a href="{}" target="_blank">View Image</a>', url)
        return ""

    view_image.short_description = "Preview"


# ========== DIC Analysis and Results ==========


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
