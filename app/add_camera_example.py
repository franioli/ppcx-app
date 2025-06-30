import os

import django

# Configure Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

# Import django models
from ppcx_app.models import (  # noqa: E402
    Camera,
    CameraCalibration,
    CameraModel,
)


def add_camera(camera_name, easting, northing, elevation=None, **kwargs):
    """
    Add a new camera to the database.

    Args:
        camera_name: Unique name for the camera
        easting: UTM easting coordinate (WGS84 UTM 32N)
        northing: UTM northing coordinate (WGS84 UTM 32N)
        elevation: Elevation in meters
        **kwargs: Additional camera properties

    Returns:
        Camera object
    """
    camera = Camera.objects.create(
        camera_name=camera_name,
        easting=easting,
        northing=northing,
        elevation=elevation,
        **kwargs,
    )
    return camera


def add_calibration(
    camera,
    calibration_date,
    colmap_model_id,
    intrinsic_params,
    image_width_px,
    image_height_px,
    is_active=True,
    **kwargs,
):
    """
    Add a new calibration for a camera.

    Args:
        camera: Camera object or camera_id
        calibration_date: Date of calibration
        colmap_model_id: ID of the COLMAP camera model
        intrinsic_params: List of intrinsic parameters for the selected model
        image_width_px: Image width in pixels
        image_height_px: Image height in pixels
        is_active: Whether this is the active calibration for the camera
        **kwargs: Additional calibration properties

    Returns:
        CameraCalibration object
    """
    if isinstance(camera, int):
        camera = Camera.objects.get(id=camera)

    # Get the model name from enum
    try:
        model_name = CameraModel(colmap_model_id).name
    except ValueError:
        model_name = f"CUSTOM_MODEL_{colmap_model_id}"

    calibration = CameraCalibration.objects.create(
        camera=camera,
        calibration_date=calibration_date,
        colmap_model_id=colmap_model_id,
        colmap_model_name=model_name,
        intrinsic_params=intrinsic_params,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        is_active=is_active,
        **kwargs,
    )

    # Ensure only one active calibration per camera
    if is_active:
        CameraCalibration.objects.filter(camera=camera, is_active=True).exclude(
            pk=calibration.pk
        ).update(is_active=False)

    return calibration


# Example usage
if __name__ == "__main__":
    # Add example camera
    camera = add_camera(
        camera_name="PlanpincieuxWide",
        easting=342000.0,
        northing=5078000.0,
        elevation=2100.0,
        model="Canon EOS 5D",
        serial_number="CANEOS12345",
        installation_date="2023-06-01",
    )

    # Add calibration
    calibration = add_calibration(
        camera=camera,
        calibration_date="2023-06-02 10:00:00",
        colmap_model_id=CameraModel.OPENCV.value,
        intrinsic_params=[2800.5, 2805.2, 2048.0, 1536.0, -0.1, 0.05, 0.001, -0.002],
        image_width_px=4096,
        image_height_px=3072,
        is_active=True,
        rotation_quaternion=[0.0, 0.0, 0.0, 1.0],
        translation_vector=[0.0, 0.0, 0.0],
    )

    print(f"Created camera: {camera}")
    print(f"Created calibration: {calibration}")
    print(f"Intrinsics: {calibration.get_intrinsics_dict()}")
