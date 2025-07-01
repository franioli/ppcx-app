import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import django
import exifread
from django.utils import timezone
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from ppcx_app.models import Camera, Image  # noqa: E402

logger = logging.getLogger("ppcx")  # Use the logger from the ppcx_app module

# Define the host and container camera directories

# Path on the host system where the script is run
HOST_CAMERA_DIR = Path("/home/fioli/storage/francesco/ppcx_db/db_import/HiRes/Wide")

# Path as seen inside the container. It MUST start with '/ppcx/fms_data/' and then the rest of the path
CONTAINER_BASE_DIR = Path("/ppcx/fms_data/Dati/HiRes/Wide")

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

SKIP_EXISTING_IMAGES = (
    True  # Set to True to skip images with the same filename already in DB
)

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


def scan_images_for_db() -> list[dict[str, Any]]:
    """
    Scan the input directory on the host system, check for valid image paths,
    convert the paths to container-relative paths, and return a list of dicts
    with all necessary data for DB population.

    Returns:
        List[dict]: Each dict contains:
            - host_path: Path to the image on the host
            - container_path: Path in the container (absolute)
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


def extract_exif_data(image_path: Path) -> dict:
    """
    Extract camera model, lens, focal length, timestamp, width, height and
    all EXIF tags (as exif_data) using exifread.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        dict: Extracted EXIF data and metadata.
    """

    def _parse_exif_date(dt_str: str) -> Any:
        try:
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except Exception:
            try:
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None

    model = lens = None
    focal = width = height = None
    timestamp = None
    exif_data = {}

    with open(image_path, "rb") as fh:
        tags = exifread.process_file(fh, details=False, builtin_types=True)
        exif_data = dict(tags)
        if "JPEGThumbnail" in exif_data:
            del exif_data["JPEGThumbnail"]

    model_tag = tags.get("Image Model")
    if isinstance(model_tag, str) and model_tag.strip():
        model = model_tag.strip()

    lens_tag = tags.get("Image LensModel") or tags.get("EXIF LensModel")
    if isinstance(lens_tag, str) and lens_tag.strip():
        lens = lens_tag.strip("\x00").strip()

    focal_tag = tags.get("EXIF FocalLength")
    if isinstance(focal_tag, (int | float)):
        focal = float(focal_tag)
    elif isinstance(focal_tag, list) and len(focal_tag) == 2 and focal_tag[1] != 0:
        focal = float(focal_tag[0]) / float(focal_tag[1])

    dt_tag = tags.get("EXIF DateTimeOriginal")
    if isinstance(dt_tag, str) and dt_tag.strip():
        parsed_dt = _parse_exif_date(dt_tag.strip())
        if parsed_dt:
            timestamp = parsed_dt

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

    return {
        "model": model,
        "lens": lens,
        "focal": focal,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "exif_data": exif_data,
    }


def process_image_entry(
    entry, cameras, existing_filenames, create_new_cameras, skip_existing_images
):
    """
    Process a single image entry: extract exif, find/create camera, and prepare DB entry.
    Returns a tuple: (status, info_dict)
    status: 'added', 'skipped', 'failed', or 'replaced'
    info_dict: dict with details for DB insertion or error logging
    """
    image_file = entry["host_path"]
    container_path = entry["container_path"]
    filename = str(container_path)

    # Fast skip or replace based on filename
    if filename in existing_filenames:
        if skip_existing_images:
            return ("skipped", {"image_file": image_file})
        else:
            # Mark for replacement (actual delete will be in main thread)
            return ("replaced", {"image_file": image_file, "filename": filename})

    # Extract EXIF data
    try:
        exif = extract_exif_data(image_file)
    except Exception as err:
        return (
            "failed",
            {"image_file": image_file, "error": f"Failed to extract EXIF: {err}"},
        )

    exif_model = exif.get("model")
    exif_lens = exif.get("lens")
    exif_focal = exif.get("focal")
    exif_timestamp = exif.get("timestamp")
    exif_width = exif.get("width")
    exif_height = exif.get("height")
    exif_data = exif.get("exif_data", {})

    db_camera_name = (
        f"{exif_model or 'Unknown'}-{exif_lens or 'Unknown'}-{int(exif_focal or 0)}"
    )

    # Find or create camera
    camera_obj = None
    for cam in cameras:
        if (
            (cam.model or "Unknown") == (exif_model or "Unknown")
            and (cam.lens or "Unknown") == (exif_lens or "Unknown")
            and int(cam.focal_length_mm or 0) == int(exif_focal or 0)
        ):
            camera_obj = cam
            break

    if camera_obj is None:
        if create_new_cameras:
            return (
                "create_camera",
                {
                    "db_camera_name": db_camera_name,
                    "exif_model": exif_model,
                    "exif_lens": exif_lens,
                    "exif_focal": exif_focal,
                    "image_file": image_file,
                    "container_path": container_path,
                    "exif_width": exif_width,
                    "exif_height": exif_height,
                    "exif_data": exif_data,
                    "exif_timestamp": exif_timestamp,
                },
            )
        else:
            return (
                "skipped",
                {"image_file": image_file, "reason": "camera does not exist"},
            )

    # Build acquisition timestamp
    if exif_timestamp:
        try:
            acquisition_timestamp = timezone.make_aware(
                exif_timestamp, timezone.get_current_timezone()
            )
        except Exception as err:
            return (
                "failed",
                {
                    "image_file": image_file,
                    "error": f"Failed to make timestamp aware: {err}",
                },
            )
    else:
        filename_stem = image_file.stem
        try:
            acquisition_timestamp = datetime.strptime(
                filename_stem, "PPCX_%Y_%m_%d_%H_%M_%S"
            )
            acquisition_timestamp = timezone.make_aware(
                acquisition_timestamp, timezone.get_current_timezone()
            )
        except ValueError:
            return (
                "failed",
                {
                    "image_file": image_file,
                    "error": "Could not extract timestamp from EXIF or filename",
                },
            )

    return (
        "add",
        {
            "camera_obj": camera_obj,
            "acquisition_timestamp": acquisition_timestamp,
            "container_path": container_path,
            "exif_width": exif_width,
            "exif_height": exif_height,
            "exif_data": exif_data,
            "filename": filename,
            "image_file": image_file,
        },
    )


def main() -> None:
    logger.info("Starting image population script.")

    images_added = 0
    images_skipped = 0
    images_failed = 0
    images_replaced = 0

    cameras = list(Camera.objects.all())
    existing_filenames = set(Image.objects.values_list("file_path", flat=True))
    image_entries = scan_images_for_db()
    logger.info(f"Found {len(image_entries)} images to process.")

    # Use ThreadPoolExecutor for parallel EXIF extraction and pre-processing
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 6) as executor:
        futures = [
            executor.submit(
                process_image_entry,
                entry,
                cameras,
                existing_filenames,
                CREATE_NEW_CAMERAS,
                SKIP_EXISTING_IMAGES,
            )
            for entry in image_entries
        ]
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Processing images"
        ):
            results.append(f.result())

    # Now handle DB operations in main thread (Django ORM is not thread-safe)
    for status, info in results:
        if status == "skipped":
            images_skipped += 1
        elif status == "replaced":
            deleted, _ = Image.objects.filter(file_path=info["filename"]).delete()
            if deleted:
                images_replaced += deleted
        elif status == "create_camera":
            try:
                camera_obj = Camera.objects.create(
                    camera_name=info["db_camera_name"],
                    model=info["exif_model"],
                    lens=info["exif_lens"],
                    focal_length_mm=info["exif_focal"],
                )
                cameras.append(camera_obj)
                logger.info(f"Created camera: {info['db_camera_name']}")
                # Now add the image
                exif_timestamp = info["exif_timestamp"]
                if exif_timestamp:
                    acquisition_timestamp = timezone.make_aware(
                        exif_timestamp, timezone.get_current_timezone()
                    )
                else:
                    filename_stem = info["image_file"].stem
                    try:
                        acquisition_timestamp = datetime.strptime(
                            filename_stem, "PPCX_%Y_%m_%d_%H_%M_%S"
                        )
                        acquisition_timestamp = timezone.make_aware(
                            acquisition_timestamp, timezone.get_current_timezone()
                        )
                    except ValueError:
                        logger.error(
                            f"Critical error: Could not extract timestamp from EXIF or filename for {info['image_file'].name}. Skipping image."
                        )
                        images_failed += 1
                        continue
                image_instance = add_image(
                    camera=camera_obj,
                    acquisition_timestamp=acquisition_timestamp,
                    file_path=str(info["container_path"]),
                    width_px=info["exif_width"],
                    height_px=info["exif_height"],
                    exif_data=info["exif_data"],
                )
                if image_instance:
                    images_added += 1
                    existing_filenames.add(str(info["container_path"]))
                else:
                    logger.error(
                        f"Failed to add image to database: {info['image_file']} - add_image returned None"
                    )
                    images_failed += 1
            except Exception as err:
                logger.error(f"Failed to create camera for {info['image_file']}: {err}")
                images_failed += 1
        elif status == "add":
            try:
                image_instance = add_image(
                    camera=info["camera_obj"],
                    acquisition_timestamp=info["acquisition_timestamp"],
                    file_path=str(info["container_path"]),
                    width_px=info["exif_width"],
                    height_px=info["exif_height"],
                    exif_data=info["exif_data"],
                )
                if image_instance:
                    images_added += 1
                    existing_filenames.add(info["filename"])
                else:
                    logger.error(
                        f"Failed to add image to database: {info['image_file']} - add_image returned None"
                    )
                    images_failed += 1
            except Exception as err:
                logger.error(f"Database error while adding {info['image_file']}: {err}")
                images_failed += 1
        elif status == "failed":
            logger.error(f"{info.get('error', 'Unknown error')} [{info['image_file']}]")
            images_failed += 1

    logger.info(
        f"Image population script finished. Added: {images_added}, Skipped: {images_skipped}, Failed: {images_failed}, Replaced: {images_replaced}"
    )


if __name__ == "__main__":
    main()
