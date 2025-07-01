import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import django
import h5py
import numpy as np
from django.db import transaction
from django.utils import timezone
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from ppcx_app.models import DIC  # noqa: E402
from ppcx_app.models import Image as ImageModel  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(
    level=logging.INFO, name="ppcx", log_to_file=True, log_folder=".logs"
)

DIC_DATA_DIR = Path("/home/fioli/storage/francesco/ppcx_db/db_import/DIC_python")

# Path on the host system where the script is run
DIC_H5_DIR_HOST = Path("/home/fioli/storage/francesco/ppcx_db/dic_results")

# Names of directories containing DIC data within each camera folder
DIC_COUPLES_DIR = "liste_coppie"
DIC_RESULTS_DIR = "matrici_spostamento"
DIC_IMAGES_DIR = "coregistrate"

DIC_RESULTS_PATTERN = "day_dic_*.txt"

CAMERA_FOLDERS = [
    # "Planpincieux_Tele",
    "Planpincieux_Wide",
]

SOFTWARE_USED = "PyLamma"

# Path as seen inside the container (it MUST match the mount in docker-compose). DO NOT CHANGE THIS!
DIC_H5_DIR_CONTAINER = Path("/ppcx/data")


def parse_couples_filename(filename: str | Path) -> datetime | None:
    """Parse couples filename to extract reference date.

    Expected format: couples_YYYYMMDD.txt
    """
    try:
        filename = Path(filename)
        if not filename.name.startswith("couples_"):
            return None

        date_str = filename.stem.split("_")[1]  # Extract YYYYMMDD part
        return datetime.strptime(date_str, "%Y%m%d")

    except (ValueError, IndexError):
        logger.error(f"Failed to parse couples filename: {filename}")
        return None


def parse_image_filename(filename: Path | str) -> datetime | None:
    """Parse image filename to extract timestamp."""
    try:
        filename = Path(filename)
        if not filename.name.startswith("PPCX_"):
            return None

        parts = filename.name.split("_")
        year, month, day, hour, minute, second = map(int, parts[2:8])
        return datetime(year, month, day, hour, minute, second)

    except (ValueError, IndexError):
        pass
    return None


def read_couples_file(couples_file: Path | str) -> list[tuple[str, str]]:
    """Read couples file and return list of image pairs."""
    couples_file = Path(couples_file)
    if not couples_file.exists():
        logger.warning(f"Couples file not found: {couples_file}")
        return []

    couples: list[tuple[str, str]] = []
    with open(couples_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                couples.append((parts[0], parts[1]))

    return couples


def find_dic_results_for_pair(
    master_img: str, slave_img: str, dic_results_dir: Path
) -> Path | None:
    """Find DIC results file for a specific image pair.

    Args:
        master_img: Master image filename
        slave_img: Slave image filename
        dic_results_dir: Directory containing DIC results files

    Returns:
        Path to DIC results file or None if not found
    """
    # Extract dates from image filenames
    master_timestamp = parse_image_filename(master_img)
    slave_timestamp = parse_image_filename(slave_img)

    if not master_timestamp or not slave_timestamp:
        return None

    # Expected DIC filename format: day_dic_YYYYMMDD-YYYYMMDD.txt
    master_date_str = master_timestamp.strftime("%Y%m%d")
    slave_date_str = slave_timestamp.strftime("%Y%m%d")

    # Try both possible combinations (master-slave and slave-master)
    possible_filenames = [
        f"day_dic_{master_date_str}-{slave_date_str}.txt",
        f"day_dic_{slave_date_str}-{master_date_str}.txt",
    ]

    for filename in possible_filenames:
        dic_file = dic_results_dir / filename
        if dic_file.exists():
            return dic_file

    return None


def read_dic_results(
    dic_file: Path | str, image: Path | str | None = None
) -> dict[str, Any] | None:
    """Read and process DIC results from a file."""
    dic_file = Path(dic_file)
    if not dic_file.exists():
        logger.error(f"DIC results file not found: {dic_file}")
        return None

    try:
        # Use numpy to read the CSV file directly
        data = np.loadtxt(dic_file, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4))

        if data.size == 0:
            return None

        # Handle case where there's only one row
        if data.ndim == 1:
            data = data.reshape(1, -1)

    except Exception as e:
        logger.error(f"Error reading DIC file {dic_file}: {e}")
        return None

    # Extract columns
    x_coords = data[:, 0].astype(int)
    y_coords = -data[:, 1].astype(int)  # Note: y component is reversed
    dx_values = data[:, 2]
    dy_values = -data[:, 3]  # Note: NS component is reversed
    magnitudes = data[:, 4]

    max_magnitude = np.max(magnitudes) if len(magnitudes) > 0 else 1.0

    # Apply bounds checking if image is provided
    if image is not None:
        try:
            img = Image.open(image)
            width, height = img.size

            # Create mask for points within bounds
            valid_mask = (
                (x_coords >= 0)
                & (x_coords < width)
                & (y_coords >= 0)
                & (y_coords < height)
            )

            # Filter arrays using the mask
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            dx_values = dx_values[valid_mask]
            dy_values = dy_values[valid_mask]
            magnitudes = magnitudes[valid_mask]

        except Exception as e:
            logger.error(f"Error processing image bounds: {e}")
            return None

    if len(x_coords) == 0:
        return None

    # Stack coordinates and vectors
    points = np.column_stack((x_coords, y_coords))
    vectors = np.column_stack((dx_values, dy_values))

    return {
        "points": points,
        "vectors": vectors,
        "magnitudes": magnitudes,
        "max_magnitude": max_magnitude,
    }


def save_dic_results_to_hdf5(dic_data: dict, file_path: str | Path) -> None:
    """
    Save DIC results to an HDF5 file in a structured way.

    The file will contain:
      - /points: Nx2 int32 array (seed_x_px, seed_y_px)
      - /vectors: Nx2 float32 array (displacement_x_px, displacement_y_px)
      - /magnitudes: Nx1 float32 array (correlation_score or magnitude)
      - /max_magnitude: scalar float32

    Args:
        dic_data (dict): Dictionary with keys 'points', 'vectors', 'magnitudes', 'max_magnitude'
        file_path (str | Path): Path to the output HDF5 file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("points", data=dic_data["points"], dtype="int32")
        f.create_dataset("vectors", data=dic_data["vectors"], dtype="float32")
        f.create_dataset("magnitudes", data=dic_data["magnitudes"], dtype="float32")
        f.create_dataset(
            "max_magnitude", data=np.array(dic_data["max_magnitude"], dtype="float32")
        )


def get_container_h5_path(host_h5_path: Path) -> str:
    """
    Convert a host h5 file path to the corresponding container path.

    Args:
        host_h5_path (Path): The path to the h5 file on the host system.

    Returns:
        str: The path as seen inside the container.
    """
    try:
        host_h5_path = host_h5_path.resolve()
        relative = host_h5_path.relative_to(DIC_H5_DIR_HOST.resolve())
        container_path = DIC_H5_DIR_CONTAINER / relative
        return str(container_path)
    except Exception as e:
        logger.error(f"Failed to convert host h5 path to container path: {e}")
        raise


def create_dic_for_pair(
    reference_date: datetime, master_img: str, slave_img: str, dic_results_dir: Path
) -> int:
    master_timestamp = parse_image_filename(master_img)
    slave_timestamp = parse_image_filename(slave_img)
    if not master_timestamp or not slave_timestamp:
        logger.error(f"Failed to parse timestamps from {master_img} or {slave_img}")
        return -1
    master_timestamp_tz = timezone.make_aware(master_timestamp)
    slave_timestamp_tz = timezone.make_aware(slave_timestamp)

    # Check if DIC already exists
    if DIC.objects.filter(
        master_timestamp=master_timestamp_tz,
        slave_timestamp=slave_timestamp_tz,
    ).exists():
        logger.debug(f"DIC already exists for pair {master_img} -> {slave_img}")
        return 0
    dic_results_file = find_dic_results_for_pair(master_img, slave_img, dic_results_dir)
    if not dic_results_file:
        logger.warning(
            f"No DIC results file found for pair {master_img} -> {slave_img}"
        )
        return -1
    try:
        with transaction.atomic():
            dic_data = read_dic_results(dic_results_file)
            if not dic_data:
                logger.warning(f"No valid DIC results in {dic_results_file}")
                return -1

            # Find related images in database
            master_name = Path(master_img).name.replace("_REG", "")
            slave_name = Path(slave_img).name.replace("_REG", "")
            master_image = ImageModel.objects.filter(
                file_path__contains=master_name
            ).first()
            slave_image = ImageModel.objects.filter(
                file_path__contains=slave_name
            ).first()
            if not master_image or not slave_image:
                logger.warning(
                    f"Related images not found for pair {master_img} -> {slave_img}"
                )
                return -1

            # Create DIC entry first to get the id
            dic = DIC.objects.create(
                reference_date=reference_date,
                master_image=master_image,
                slave_image=slave_image,
                master_timestamp=master_timestamp_tz,
                slave_timestamp=slave_timestamp_tz,
                result_file_path="",  # Temporary, will update after saving h5
                software_used="PyLamma",
            )

            # Compose HDF5 file path (host) with DIC id
            h5_file_name = f"{dic.pk:08d}_{reference_date.strftime('%Y%m%d')}_{Path(master_img).stem}_{Path(slave_img).stem}.h5"
            save_dic_results_to_hdf5(dic_data, DIC_H5_DIR_HOST / h5_file_name)

            # Convert to container path for DB and store it
            h5_file_path_container = get_container_h5_path(
                DIC_H5_DIR_HOST / h5_file_name
            )
            dic.result_file_path = h5_file_path_container
            dic.save(update_fields=["result_file_path"])

            logger.debug(
                f"Successfully added DIC for pair {master_img} -> {slave_img} (date: {reference_date.strftime('%Y-%m-%d')})"
            )
            return 1
    except Exception as e:
        logger.error(f"Error creating DIC for pair {master_img} -> {slave_img}: {e}")
        return -1


def process_couples_file(
    couples_file: Path, dic_results_dir: Path
) -> tuple[int, int, int]:
    reference_date = parse_couples_filename(couples_file)
    if not reference_date:
        logger.error(f"Failed to parse reference date from {couples_file}")
        return 0, 0, 1
    couples = read_couples_file(couples_file)
    if not couples:
        logger.warning(f"No couples found in {couples_file}")
        return 0, 0, 0
    logger.debug(
        f"Processing {len(couples)} pairs for date {reference_date.strftime('%Y-%m-%d')}"
    )
    added = skipped = failed = 0
    for master_img, slave_img in couples:
        result = create_dic_for_pair(
            reference_date, master_img, slave_img, dic_results_dir
        )
        if result == 1:
            added += 1
        elif result == 0:
            skipped += 1
        elif result == -1:
            failed += 1
    return added, skipped, failed


def main():
    logger.info("Starting DIC data population (couples-first approach)...")
    total_added = total_skipped = total_failed = 0
    total_couples_files = 0
    year_cam_couples = []
    for year_dir in sorted(DIC_DATA_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.startswith("20"):
            continue
        for cam in CAMERA_FOLDERS:
            cam_dir = year_dir / cam
            required_dirs = [
                cam_dir / DIC_COUPLES_DIR,
                cam_dir / DIC_RESULTS_DIR,
                cam_dir / DIC_IMAGES_DIR,
            ]
            if not all(d.is_dir() for d in required_dirs):
                continue
            couples_dir, dic_results_dir, _ = required_dirs
            couples_files = sorted(couples_dir.glob("couples_*.txt"))
            year_cam_couples.append((year_dir, cam, couples_files, dic_results_dir))
            total_couples_files += len(couples_files)
    logger.info(
        f"Found {total_couples_files} couples files to process across "
        f"{len(year_cam_couples)} year/camera combinations"
    )
    for year_dir, cam, couples_files, dic_results_dir in year_cam_couples:
        if not couples_files:
            continue
        logger.info(
            f"Processing {year_dir.name}/{cam} with {len(couples_files)} couples files"
        )
        with tqdm(couples_files, desc=f"{year_dir.name}/{cam}") as pbar:
            for couples_file in pbar:
                try:
                    added, skipped, failed = process_couples_file(
                        couples_file, dic_results_dir
                    )
                    total_added += added
                    total_skipped += skipped
                    total_failed += failed
                    pbar.set_postfix(
                        added=total_added,
                        skipped=total_skipped,
                        failed=total_failed,
                    )
                except Exception as e:
                    logger.error(f"Error processing couples file {couples_file}: {e}")
                    total_failed += 1
                    pbar.set_postfix(
                        added=total_added,
                        skipped=total_skipped,
                        failed=total_failed,
                    )
    logger.info(
        f"DIC data population finished. "
        f"Added: {total_added}, Skipped: {total_skipped}, Failed: {total_failed}"
    )


if __name__ == "__main__":
    logger.info("Starting DIC data population script (couples-first approach)...")
    main()
