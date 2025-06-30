import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import django
import exifread
from django.utils import timezone
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from ppcx_app.models import Camera, Image  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(
    level=logging.INFO, name="ppcx", log_to_file=True, log_folder=".logs"
)

# Define the host and container camera directories

# Path on the host system where the script is run
HOST_CAMERA_DIR = Path("/home/fioli/storage/francesco/ppcx_db/db_import/HiRes/Tele")

# Path as seen inside the container. DO NOT CHANGE THIS (unless the source images have been moved to another place)!
CONTAINER_BASE_DIR = Path("/ppcx/fms_data/Dati/HiRes")

# CAMERA_DIR = Path("/home/fioli/storage/fms/Dati/HiRes/Wide")
CAMERA_DIR = HOST_CAMERA_DIR
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


def scan_images_for_db():
    """
    Scan the input directory on the host system, check for valid image paths,
    convert the paths to container-relative paths, and return a list of dicts
    with all necessary data for DB population.
    Returns:
        List[dict]: Each dict contains:
            - host_path: Path to the image on the host
            - container_rel_path: Path relative to CONTAINER_BASE_DIR
    """
    image_entries = []
    for year_item in sorted(HOST_CAMERA_DIR.iterdir()):
        if not (year_item.is_dir() and year_item.name.isdigit()):
            continue
        year = int(year_item.name)
        if not (2000 < year < 2100):
            continue
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
                        image_file.is_file()
                        and image_file.suffix.lower() in IMAGE_EXTENSIONS
                    ):
                        try:
                            container_path = Path(get_container_path(image_file))
                            image_entries.append(
                                {
                                    "host_path": image_file,
                                    "container_path": container_path,
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Skipping {image_file}: {e}")
    return image_entries


def get_container_path(host_path: Path) -> str:
    """
    Convert a host path to the corresponding container path.
    Only the base path in the container is set; the rest is built automatically.
    Checks if the corresponding file exists in the container. If not, exits the script.
    """
    try:
        rel_path = host_path.relative_to(HOST_CAMERA_DIR)
    except ValueError:
        logger.error(
            f"Image path {host_path} is not under HOST_CAMERA_DIR {HOST_CAMERA_DIR}"
        )
        sys.exit(1)
    container_path = CONTAINER_BASE_DIR / rel_path
    return str(container_path)


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


def main():
    logger.info("Starting image population script.")

    images_added = 0
    images_skipped = 0
    images_failed = 0

    # Preload all cameras in a dict for fast lookup
    camera_dict = {c.camera_name: c for c in Camera.objects.all()}

    # Preload all existing images (camera_id, acquisition_timestamp) for fast lookup
    existing_images = set(
        Image.objects.values_list("camera_id", "acquisition_timestamp")
    )

    # Use the new scan_images_for_db function
    image_entries = scan_images_for_db()
    logger.info(f"Found {len(image_entries)} images to process.")

    for entry in tqdm(image_entries, desc="Processing images"):
        image_file = entry["host_path"]
        container_path = entry["container_path"]
        try:
            # Infer EXIF data
            exif = extract_exif_data(image_file)
            exif_model = exif.get("model")
            exif_lens = exif.get("lens")
            exif_focal = exif.get("focal")
            exif_timestamp = exif.get("timestamp")
            exif_width = exif.get("width")
            exif_height = exif.get("height")
            exif_data = exif.get("exif_data", {})

            db_camera_name = f"{exif_model or 'Unknown'}-{exif_lens or 'Unknown'}-{int(exif_focal or 0)}"
            camera_obj = camera_dict.get(db_camera_name)

            if camera_obj is None:
                if CREATE_NEW_CAMERAS:
                    camera_obj = Camera.objects.create(
                        camera_name=db_camera_name,
                        model=exif_model,
                        lens=exif_lens,
                        focal_length_mm=exif_focal,
                    )
                    camera_dict[db_camera_name] = camera_obj
                    logger.info(f"Created camera: {db_camera_name}")
                else:
                    logger.warning(
                        f"Skipping image {image_file} - camera does not exist and CREATE_NEW_CAMERAS is False."
                    )
                    images_skipped += 1
                    continue

            # Build acquisition timestamp
            if exif_timestamp:
                acquisition_timestamp = timezone.make_aware(
                    exif_timestamp, timezone.get_current_timezone()
                )
            else:
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

            # Differential: skip if already in DB
            if (
                camera_obj.pk,
                acquisition_timestamp,
            ) in existing_images and not FORCE_UPDATE:
                images_skipped += 1
                continue

            # Create the DB entry
            try:
                image_instance = add_image(
                    camera=camera_obj,
                    acquisition_timestamp=acquisition_timestamp,
                    file_path=str(container_path),
                    width_px=exif_width,
                    height_px=exif_height,
                    exif_data=exif_data,
                )

                if image_instance:
                    images_added += 1
                    existing_images.add((camera_obj.pk, acquisition_timestamp))
                else:
                    logger.error(
                        f"Failed to add image to database: {image_file} - add_image returned None"
                    )
                    images_failed += 1

            except Exception as err:
                logger.error(f"Database error while adding {image_file}: {err}")
                images_failed += 1

        except Exception as err:
            logger.error(f"Error processing image {image_file}: {err}")
            images_failed += 1

    logger.info(
        f"Image population script finished. Added: {images_added}, Skipped: {images_skipped}, Failed: {images_failed}"
    )


if __name__ == "__main__":
    main()
