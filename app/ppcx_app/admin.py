import json

from django import forms
from django.contrib import admin
from django.contrib.gis import admin as gis_admin
from django.contrib.gis.forms import OSMWidget
from django.db import models
from django.urls import reverse
from django.utils.html import format_html

from .models import DIC, Camera, CameraCalibration, Image

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
        "id",
        "camera_name",
        "model",
        "lens",
        "focal_length_mm",
        "installation_date",
        "image_count",
        "min_image_date",
        "max_image_date",
        "image_count_link",
    )
    search_fields = ("camera_name", "serial_number")
    list_filter = ("installation_date",)
    readonly_fields = ("id",)  # Always show the id in the change form

    def image_count(self, obj):
        """Display the number of images for this camera"""
        return obj.images.count()

    def min_image_date(self, obj):
        """Display the earliest image date for this camera"""
        min_date = obj.images.aggregate(min_date=models.Min("acquisition_timestamp"))[
            "min_date"
        ]
        return min_date.strftime("%Y-%m-%d %H:%M") if min_date else "No images"

    def max_image_date(self, obj):
        """Display the latest image date for this camera"""
        max_date = obj.images.aggregate(max_date=models.Max("acquisition_timestamp"))[
            "max_date"
        ]
        return max_date.strftime("%Y-%m-%d %H:%M") if max_date else "No images"

    def get_queryset(self, request):
        """Optimize queries by prefetching related images"""
        queryset = super().get_queryset(request)
        return queryset.prefetch_related("images")

    def image_count_link(self, obj):
        """Show count with link to images"""
        count = obj.images.count()
        if count > 0:
            url = reverse("admin:ppcx_app_image_changelist")
            return format_html(
                '<a href="{}?camera__id__exact={}">{} images</a>', url, obj.pk, count
            )
        return "0 images"


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
    readonly_fields = ("id",)  # Always show the id in the change form

    def save_model(self, request, obj, form, change):
        # Ensure only one active calibration per camera
        if obj.is_active:
            CameraCalibration.objects.filter(camera=obj.camera, is_active=True).exclude(
                pk=obj.pk
            ).update(is_active=False)
        super().save_model(request, obj, form, change)


# ========== Filters based on datetime ==========


class BaseDateFilter(admin.SimpleListFilter):
    """Base class for date filters that can be used with any model"""

    date_field = None  # To be overridden in subclasses

    @classmethod
    def create(cls, date_field):
        """Factory method to create a filter for a specific date field"""
        return type(f"{cls.__name__}_{date_field}", (cls,), {"date_field": date_field})


class YearFilterBase(BaseDateFilter):
    title = "year"
    parameter_name = "year"

    def lookups(self, request, model_admin):
        # Get all unique years from the specified date field
        years = model_admin.model.objects.dates(self.date_field, "year").values_list(
            f"{self.date_field}__year", flat=True
        )
        return [(str(year), str(year)) for year in sorted(years, reverse=True)]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(**{f"{self.date_field}__year": self.value()})


class MonthFilterBase(BaseDateFilter):
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
            return queryset.filter(**{f"{self.date_field}__month": self.value()})


class DayFilterBase(BaseDateFilter):
    title = "day"
    parameter_name = "day"

    def lookups(self, request, model_admin):
        return [(str(i), str(i)) for i in range(1, 32)]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(**{f"{self.date_field}__day": self.value()})


class TimeOfDayFilterBase(BaseDateFilter):
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
            return queryset.filter(**{f"{self.date_field}__hour": hour})


# ========== Images ==========

# Create filters for Image model (acquisition_timestamp)
ImageYearFilter = YearFilterBase.create("acquisition_timestamp")
ImageMonthFilter = MonthFilterBase.create("acquisition_timestamp")
ImageDayFilter = DayFilterBase.create("acquisition_timestamp")
ImageTimeOfDayFilter = TimeOfDayFilterBase.create("acquisition_timestamp")


class PreviewWidget(forms.TextInput):
    """
    A simple widget that renders the text input for file_path plus a small image preview
    when an image URL is provided (set in the form __init__).
    """

    def __init__(self, attrs=None, image_url: str | None = None):
        super().__init__(attrs)
        self.image_url = image_url

    def render(self, name, value, attrs=None, renderer=None):
        input_html = super().render(name, value, attrs=attrs, renderer=renderer)
        if self.image_url:
            img_html = format_html(
                '<div style="margin-top:6px;"><img src="{}" style="max-width:220px; max-height:140px; '
                'border:1px solid #ddd; border-radius:4px;" alt="preview" /></div>',
                self.image_url,
            )
            return format_html("{}{}", input_html, img_html)
        return input_html


class ImageAdminForm(forms.ModelForm):
    class Meta:
        model = Image
        exclude = (
            "exif_data",
        )  # exclude raw exif_data so the admin form does not try to render it
        widgets = {}  # keep other widgets as needed (exif_data removed)

    def __init__(self, *args, **kwargs):
        # instance may be None for add forms
        instance = kwargs.get("instance")
        super().__init__(*args, **kwargs)

        # Pretty-format EXIF for the readonly formatted_exif_data display (if present)
        if instance and getattr(instance, "exif_data", None):
            try:
                formatted_json = json.dumps(
                    instance.exif_data, indent=2, ensure_ascii=False, sort_keys=True
                )
                self.formatted_exif_initial = formatted_json
            except (TypeError, ValueError):
                self.formatted_exif_initial = None

        # Replace the file_path widget with a PreviewWidget when we have an instance with id
        preview_url = None
        try:
            if instance and instance.pk:
                preview_url = reverse("serve_image", args=[instance.pk])
        except Exception:
            preview_url = None

        if "file_path" in self.fields:
            self.fields["file_path"].widget = PreviewWidget(
                attrs={"size": 60}, image_url=preview_url
            )


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "camera",
        "acquisition_timestamp",
        "camera__model",
        "camera__focal_length_mm",
        "file_name",
        "view_image",
    )
    list_filter = [
        "camera",
        "camera__focal_length_mm",
        "acquisition_timestamp",
        ImageYearFilter,
        ImageMonthFilter,
        ImageDayFilter,
        ImageTimeOfDayFilter,
    ]
    search_fields = ["id", "camera__camera_name", "file_path"]
    date_hierarchy = "acquisition_timestamp"
    # Always show the id in the change form and keep formatted_exif_data readonly
    readonly_fields = ("id", "formatted_exif_data")
    form = ImageAdminForm

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

    def view_image(self, obj):
        if obj.file_path and obj.id:
            url = reverse("serve_image", args=[obj.id])
            return format_html('<a href="{}" target="_blank">View Image</a>', url)
        return ""


# ========== DIC Analysis and Results ==========

# Create filters for DIC model (master_timestamp)
DICYearFilter = YearFilterBase.create("master_timestamp")
DICMonthFilter = MonthFilterBase.create("master_timestamp")
DICDayFilter = DayFilterBase.create("master_timestamp")
DICTimeOfDayFilter = TimeOfDayFilterBase.create("master_timestamp")


@admin.register(DIC)
class DICAdmin(admin.ModelAdmin):
    show_full_result_count = False
    list_display = [
        "id",
        "reference_date",
        "master_image__camera",
        "master_timestamp",
        "slave_timestamp",
        "master_image",
        "slave_image",
        "time_difference_hours",
        "visualize_dic",
        "get_data",
        "download_csv",
        "download_quiver",
    ]
    inlines = []
    list_filter = [
        "master_timestamp",
        "master_image__camera__camera_name",
        DICYearFilter,
        DICMonthFilter,
        DICDayFilter,
        "time_difference_hours",
    ]
    search_fields = ["id", "master_timestamp", "time_difference_hours"]
    date_hierarchy = "reference_date"
    readonly_fields = ("id",)  # Always show the id in the change form

    def visualize_dic(self, obj):
        """Link to visualize DIC results"""
        if obj.result_file_path:
            url = reverse("visualize_dic", args=[obj.id])
            return format_html('<a href="{}" target="_blank">Visualize</a>', url)
        return "No visualization available"

    def get_data(self, obj):
        """Link to view DIC HDF5 data"""
        if obj.result_file_path:
            url = reverse("serve_dic_h5", args=[obj.id])
            return format_html('<a href="{}" target="_blank">Get data</a>', url)
        return "No data available"

    def download_csv(self, obj):
        """Link to view DIC HDF5 data"""
        if obj.result_file_path:
            url = reverse("serve_dic_h5_as_csv", args=[obj.id])
            return format_html('<a href="{}" target="_blank">Download CSV</a>', url)
        return "No data available"

    def download_quiver(self, obj):
        """Link to download OpenCV quiver PNG for this DIC"""
        if obj.result_file_path:
            url = reverse("serve_dic_quiver", args=[obj.id])
            return format_html('<a href="{}" target="_blank">Download Quiver</a>', url)
        return "No data available"
