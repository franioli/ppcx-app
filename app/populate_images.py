import logging
import os
from datetime import datetime
from pathlib import Path

import django
import exifread
from django.utils import timezone
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from glacier_monitoring_app.models import Camera, Image  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(
    level=logging.INFO, name="ppcx", log_to_file=True, log_folder=".logs"
)

CAMERA_DIR = Path("/data/Dati/HiRes/Wide")
IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")

MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
}

CREATE_NEW_CAMERAS = True  # Set to True to create new cameras if they don't exist

FORCE_UPDATE = False  # Set to True to force update of existing images

# --- End Configuration ---


def add_image(camera, acquisition_timestamp, file_path, **kwargs):
    """
    Add a new image to the database.

    Args:
        camera: Camera object or camera_id
        acquisition_timestamp: Timestamp when the image was acquired
        file_path: Path to the image file
        **kwargs: Additional image properties

    Returns:
        Image object
    """
    if isinstance(camera, int):
        camera = Camera.objects.get(id=camera)

    image = Image.objects.create(
        camera=camera,
        acquisition_timestamp=acquisition_timestamp,
        file_path=file_path,
        **kwargs,
    )
    return image


def parse_month(month_str):
    m = month_str.strip().lower()
    if m in MONTH_NAME_TO_NUMBER:
        return MONTH_NAME_TO_NUMBER[m]
    try:
        month_num = int(month_str)
        if 1 <= month_num <= 12:
            return month_num
    except ValueError:
        pass
    logger.warning(f"Could not parse month: '{month_str}'")
    return None


def extract_exif_data(image_path):
    """
    Extract camera model, lens, focal length, timestamp, width, height and
    all EXIF tags (as exif_data) using exifread.
    """

    def _parse_exif_date(dt_str):
        try:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except Exception:
            try:
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                logger.error(f"Could not parse EXIF date: '{dt_str}' from {image_path}")
                return None

    # Defaults
    model = lens = None
    focal = width = height = None
    timestamp = None
    exif_data = {}

    # Parse with exifread
    with open(image_path, "rb") as fh:
        tags = exifread.process_file(fh, details=False, builtin_types=True)
        exif_data = dict(tags)  # Keep all tags for further reference
        # Remove the JPEGThumbnail tag if it exists
        if "JPEGThumbnail" in exif_data:
            del exif_data["JPEGThumbnail"]

    # Model
    model_tag = tags.get("Image Model")
    if isinstance(model_tag, str) and model_tag.strip():
        model = model_tag.strip()

    # Lens
    lens_tag = tags.get("Image LensModel") or tags.get("EXIF LensModel")
    if isinstance(lens_tag, str) and lens_tag.strip():
        lens = lens_tag.strip("\x00").strip()

    # Focal length (could be a float or a ratio)
    focal_tag = tags.get("EXIF FocalLength")
    if isinstance(focal_tag, int | float):
        focal = float(focal_tag)
    elif isinstance(focal_tag, list) and len(focal_tag) == 2 and focal_tag[1] != 0:
        focal = float(focal_tag[0]) / float(focal_tag[1])

    # Timestamp
    dt_tag = tags.get("EXIF DateTimeOriginal")
    if isinstance(dt_tag, str) and dt_tag.strip():
        parsed_dt = _parse_exif_date(dt_tag.strip())
        if parsed_dt:
            timestamp = parsed_dt

    # Width and height
    width_tag = tags.get("EXIF ExifImageWidth")
    height_tag = tags.get("EXIF ExifImageLength")
    if isinstance(width_tag, int):
        width = width_tag
    elif isinstance(width_tag, list) and len(width_tag) == 2 and width_tag[1] != 0:
        width = width_tag[0] / width_tag[1]

    if isinstance(height_tag, int):
        height = height_tag
    elif isinstance(height_tag, list) and len(height_tag) == 2 and height_tag[1] != 0:
        height = height_tag[0] / height_tag[1]

    # Log only critical missing info
    if not timestamp:
        logger.error(
            f"No timestamp extracted from '{image_path}' - this is required for processing."
        )

    # Log other missing info at debug level or only if specifically needed
    if not model:
        logger.debug(f"No camera model extracted from '{image_path}'.")
    if not lens:
        logger.debug(f"No lens info extracted from '{image_path}'.")
    if not focal:
        logger.debug(f"No focal length extracted from '{image_path}'.")
    if not width or not height:
        logger.debug(f"No image dimensions extracted from '{image_path}'.")

    return {
        "model": model,
        "lens": lens,
        "focal": focal,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "exif_data": exif_data,
    }


def populate_images():
    logger.info("Starting image population script...")

    images_processed = 0
    images_added = 0
    images_skipped = 0
    images_failed = 0

    # Get all year directories for progress bar
    year_dirs = [
        year_item
        for year_item in sorted(CAMERA_DIR.iterdir())
        if year_item.is_dir() and year_item.name.isdigit()
    ]

    # Filter valid years
    valid_year_dirs = []
    for year_item in year_dirs:
        try:
            year = int(year_item.name)
            if 2000 < year < 2100:
                valid_year_dirs.append(year_item)
        except ValueError:
            continue

    for year_item in tqdm(valid_year_dirs, desc="Processing years", unit="year"):
        year = int(year_item.name)
        logger.info(f"Processing year: {year}")

        for month_item in sorted(year_item.iterdir()):
            if not month_item.is_dir():
                continue
            month = parse_month(month_item.name)
            if not month:
                continue

            for day_item in sorted(month_item.iterdir()):
                if not day_item.is_dir():
                    continue

                for image_file in sorted(day_item.iterdir()):
                    if (
                        not image_file.is_file()
                        or image_file.suffix.lower() not in IMAGE_EXTENSIONS
                    ):
                        continue

                    # Infer EXIF data
                    exif = extract_exif_data(image_file)
                    exif_model = exif.get("model")
                    exif_lens = exif.get("lens")
                    exif_focal = exif.get("focal")
                    exif_timestamp = exif.get("timestamp")
                    exif_width = exif.get("width")
                    exif_height = exif.get("height")
                    exif_data = exif.get("exif_data", {})

                    # Check if a Camera object exists for this image. Check by model and lens and focal length
                    db_camera_name = f"{exif_model or 'Unknown'}-{exif_lens or 'Unknown'}-{int(exif_focal or 0)}"
                    camera_obj = Camera.objects.filter(
                        camera_name=db_camera_name
                    ).first()

                    # If no camera is found yet, create one if user agrees
                    if camera_obj is None:
                        if CREATE_NEW_CAMERAS:
                            camera_name = f"{exif_model or 'Unknown'}-{exif_lens or 'Unknown'}-{int(exif_focal or 0)}"
                            camera_obj = Camera.objects.create(
                                camera_name=camera_name,
                                model=exif_model,
                                lens=exif_lens,
                                focal_length_mm=exif_focal,
                            )
                            logger.info(f"Created camera: {camera_name}")
                        else:
                            logger.warning(
                                f"Skipping image {image_file} - camera does not exist and CREATE_NEW_CAMERAS is False."
                            )
                            images_skipped += 1
                            continue

                    #  Build acquisition timestamp
                    if exif_timestamp:
                        acquisition_timestamp = timezone.make_aware(
                            exif_timestamp, timezone.get_current_timezone()
                        )

                    else:
                        # Build timestamp from the filename structure
                        # PPCX_2013_08_28_14_00_02.jpg
                        filename = image_file.stem
                        try:
                            acquisition_timestamp = datetime.strptime(
                                filename, "PPCX_%Y_%m_%d_%H_%M_%S"
                            )
                            acquisition_timestamp = timezone.make_aware(
                                acquisition_timestamp, timezone.get_current_timezone()
                            )
                        except ValueError:
                            logger.error(
                                f"Critical error: Could not extract timestamp from EXIF or filename for {image_file.name}. Skipping image."
                            )
                            images_failed += 1
                            continue

                    # Check if the image already exists in the database.
                    existing_image = Image.objects.filter(
                        camera=camera_obj,
                        acquisition_timestamp=acquisition_timestamp,
                    ).first()
                    if existing_image and not FORCE_UPDATE:
                        logger.debug(
                            f"Image already exists in DB: {image_file} (ID: {existing_image.pk})"
                        )
                        images_skipped += 1
                        continue
                    elif existing_image and FORCE_UPDATE:
                        #  If FORCE_UPDATE is True, delete the existing image
                        logger.info(
                            f"Updating existing image: {image_file} (ID: {existing_image.pk})"
                        )
                        existing_image.delete()
                        existing_image = None

                    # Create the DB entry
                    images_processed += 1
                    try:
                        image_instance = add_image(
                            camera=camera_obj,
                            acquisition_timestamp=acquisition_timestamp,
                            file_path=str(image_file),
                            width_px=exif_width,
                            height_px=exif_height,
                            exif_data=exif_data,
                        )

                        if image_instance:
                            logger.info(
                                f"Successfully added image: {image_file} (ID: {image_instance.pk})"
                            )
                            images_added += 1
                        else:
                            logger.error(
                                f"Failed to add image to database: {image_file} - add_image returned None"
                            )
                            images_failed += 1

                    except Exception as err:
                        logger.error(f"Database error while adding {image_file}: {err}")
                        images_failed += 1

    logger.info(
        f"Image population script finished. Processed: {images_processed}, Added: {images_added}, Skipped: {images_skipped}, Failed: {images_failed}"
    )


if __name__ == "__main__":
    populate_images()
