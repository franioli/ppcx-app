import os

import django

# Configure Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from glacier_monitoring_app.models import DICAnalysis, DICResult, Image  # noqa: E402


def add_dic_analysis(reference_image, secondary_image, **kwargs):
    """
    Create a new DIC analysis between two images.

    Args:
        reference_image: Reference image object or image_id
        secondary_image: Secondary image object or image_id
        **kwargs: Additional analysis properties

    Returns:
        DICAnalysis object
    """
    if isinstance(reference_image, int):
        reference_image = Image.objects.get(id=reference_image)

    if isinstance(secondary_image, int):
        secondary_image = Image.objects.get(id=secondary_image)

    analysis = DICAnalysis.objects.create(
        reference_image=reference_image, secondary_image=secondary_image, **kwargs
    )
    return analysis


def add_dic_result(analysis, seed_x_ref_px, seed_y_ref_px, **kwargs):
    """
    Add a result point from a DIC analysis.

    Args:
        analysis: DICAnalysis object or analysis_id
        seed_x_ref_px: X-coordinate of seed point in reference image
        seed_y_ref_px: Y-coordinate of seed point in reference image
        **kwargs: Additional result properties

    Returns:
        DICResult object
    """
    if isinstance(analysis, int):
        analysis = DICAnalysis.objects.get(id=analysis)

    result = DICResult.objects.create(
        analysis=analysis,
        seed_x_ref_px=seed_x_ref_px,
        seed_y_ref_px=seed_y_ref_px,
        **kwargs,
    )
    return result
