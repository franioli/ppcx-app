import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import django
import numpy as np
from django.db import transaction
from django.utils import timezone
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from glacier_monitoring_app.models import DICAnalysis, DICResult  # noqa: E402
from glacier_monitoring_app.models import Image as ImageModel  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(
    level=logging.INFO, name="ppcx", log_to_file=True, log_folder=".logs"
)


DIC_DATA_DIR = Path("/home/francesco/cnr/fms/DIC_python")
DIC_COUPLES_DIR = "liste_coppie"
DIC_RESULTS_DIR = "matrici_spostamento"
DIC_IMAGES_DIR = "coregistrate"

DIC_RESULTS_PATTERN = "day_dic_*.txt"

CAMERA_FOLDERS = [
    "Planpincieux_Tele",
    "Planpincieux_Wide",
]


def parse_day_dic_filename(
    filename: str | Path,
) -> tuple[datetime, datetime] | tuple[None, None]:
    parts = Path(filename).stem.split("_")
    if len(parts) < 3 or parts[0] != "day" or parts[1] != "dic":
        logger.error(f"Filename does not match expected format: {filename}")
        return None, None

    cur_day_str, prev_day_str = parts[2].split("-")
    cur_day = datetime.strptime(cur_day_str, "%Y%m%d")
    prev_day = datetime.strptime(prev_day_str, "%Y%m%d")

    return cur_day, prev_day


def parse_image_filename(filename: Path | str) -> datetime | None:
    try:
        filename = Path(filename)
        if not filename.name.startswith("PPCX_"):
            return None

        parts = filename.name.split("_")
        year, month, day, hour, minute = map(int, parts[2:7])
        return datetime(year, month, day, hour, minute)

    except (ValueError, IndexError):
        pass
    return None


def read_couples_file(couples_file: Path | str) -> list[tuple[str, str]]:
    couples_file = Path(couples_file)
    if not couples_file.exists():
        print(f"Couples file not found: {couples_file}")
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


def read_dic_results(
    dic_file: Path | str, image: Path | str | None = None
) -> dict[str, Any] | None:
    """Read and process DIC results for an image from a DIC results file.

    Args:
        dic_file: Path to the DIC results file.
        image: Optional path to an image file for bounds checking.

    Returns:
        Dict with points, vectors, magnitudes, and max_magnitude, or None if no valid points or file not found.
    """
    dic_file = Path(dic_file)
    if not dic_file.exists():
        logger.error(f"DIC results file not found: {dic_file}")
        return None

    try:
        # Use numpy to read the CSV file directly
        # Assuming columns are: X, Y, EW, NS, V (adjust usecols if different)
        data = np.loadtxt(dic_file, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4))

        if data.size == 0:
            return None

        # Handle case where there's only one row (numpy returns 1D array)
        if data.ndim == 1:
            data = data.reshape(1, -1)

    except Exception as e:
        logger.error(f"Error reading DIC file: {e}")
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


def get_dic_metadata(daily_dic: Path, couples_dir: Path) -> dict | None:
    """Process a single DIC file and return analysis data."""
    cur_day, prev_day = parse_day_dic_filename(daily_dic.name)
    if not cur_day or not prev_day:
        return None

    couples_file = couples_dir / f"couples_{cur_day.strftime('%Y%m%d')}.txt"
    couples = read_couples_file(couples_file)
    if not couples:
        return None

    # Use first couple for analysis
    master_img, slave_img = couples[0]
    master_timestamp = parse_image_filename(master_img)
    slave_timestamp = parse_image_filename(slave_img)
    if not master_timestamp or not slave_timestamp:
        return None

    return {
        "master_img": master_img,
        "slave_img": slave_img,
        "master_timestamp": timezone.make_aware(master_timestamp),
        "slave_timestamp": timezone.make_aware(slave_timestamp),
    }


def create_dic_analysis(daily_dic, couples_dir) -> int:
    """Create DIC analysis and results in database.

    Returns:
        1: Successfully added
        0: Skipped (already exists or no data)
        -1: Failed (error during processing)
    """
    logger.debug(f"Processing DIC file: {daily_dic.name}")

    analysis_data = get_dic_metadata(daily_dic, couples_dir)
    if not analysis_data:
        logger.error(f"Failed to get metadata for {daily_dic}")
        return -1  # Failed - no metadata

    # Check if analysis already exists
    if DICAnalysis.objects.filter(
        master_timestamp=analysis_data["master_timestamp"],
        slave_timestamp=analysis_data["slave_timestamp"],
    ).exists():
        logger.debug(f"Analysis already exists for {daily_dic.name}")
        return 0  # Skipped - already exists

    try:
        # Use atomic transaction to ensure all-or-nothing database operations
        with transaction.atomic():
            # Read DIC results
            dic_results = read_dic_results(daily_dic)
            if not dic_results:
                logger.warning(f"No DIC results found for {daily_dic.name}")
                return -1  # Failed - no results

            # Find related images
            master_name = Path(analysis_data["master_img"]).name.replace("_REG", "")
            slave_name = Path(analysis_data["slave_img"]).name.replace("_REG", "")
            master_image = ImageModel.objects.filter(
                file_path__contains=master_name
            ).first()
            slave_image = ImageModel.objects.filter(
                file_path__contains=slave_name
            ).first()

            # Create DIC analysis
            dic_analysis = DICAnalysis.objects.create(
                master_timestamp=analysis_data["master_timestamp"],
                slave_timestamp=analysis_data["slave_timestamp"],
                master_image=master_image if master_image else None,
                slave_image=slave_image if slave_image else None,
                software_used="PyLamma",
            )

            # Create DIC results in bulk
            dic_result_entries = []
            for point, vector, magnitude in zip(
                dic_results["points"],
                dic_results["vectors"],
                dic_results["magnitudes"],
                strict=False,
            ):
                dic_result_entries.append(
                    DICResult(
                        analysis=dic_analysis,
                        seed_x_px=int(point[0]),
                        seed_y_px=int(point[1]),
                        displacement_x_px=float(vector[0]),
                        displacement_y_px=float(vector[1]),
                        displacement_magnitude_px=float(magnitude),
                    )
                )

            if dic_result_entries:
                DICResult.objects.bulk_create(dic_result_entries)
                logger.debug(
                    f"Successfully added {len(dic_result_entries)} DIC results for {daily_dic.name}"
                )

                # If we reach here, all operations succeeded and will be committed
                return 1  # Success
            else:
                # No results to create, rollback transaction
                transaction.set_rollback(True)
                return -1  # Failed - no results to create

    except Exception as e:
        # Transaction will automatically rollback on any exception
        logger.error(f"Error creating DIC analysis for {daily_dic.name}: {e}")
        return -1  # Failed - error during processing

    return 0  # Skipped - no results to create


def populate_dic_data():
    logger.info("Fetching DIC data...")
    dic_data_added = 0
    dic_data_skipped = 0
    dic_data_failed = 0

    # First pass: count total files to process
    total_files = 0
    year_cam_files = []
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

            couples_dir, dic_res_dir, _ = required_dirs
            dic_files = list(dic_res_dir.glob(DIC_RESULTS_PATTERN))
            year_cam_files.append((year_dir, cam, couples_dir, dic_files))
            total_files += len(dic_files)
    logger.info(
        f"Found {total_files} DIC files to process across {len(year_cam_files)} year/camera combinations"
    )

    logger.info("Populating database...")
    for year_dir, cam, couples_dir, dic_files in year_cam_files:
        logger.info(f"Processing {year_dir.name}/{cam} with {len(dic_files)} files")

        with tqdm(dic_files, desc=f"{year_dir.name}/{cam}") as pbar:
            for daily_dic in pbar:
                try:
                    result = create_dic_analysis(daily_dic, couples_dir)

                    if result == 1:
                        dic_data_added += 1
                    elif result == 0:
                        dic_data_skipped += 1
                    elif result == -1:
                        dic_data_failed += 1

                    # Show statistics in progress bar
                    pbar.set_postfix(
                        added=dic_data_added,
                        skipped=dic_data_skipped,
                        failed=dic_data_failed,
                    )

                except Exception as e:
                    logger.error(f"Error processing {daily_dic.name}: {e}")
                    dic_data_failed += 1
                    pbar.set_postfix(
                        added=dic_data_added,
                        skipped=dic_data_skipped,
                        failed=dic_data_failed,
                    )

    logger.info(
        f"DIC data population finished. Added: {dic_data_added}, Skipped: {dic_data_skipped}, Failed: {dic_data_failed}"
    )


if __name__ == "__main__":
    logger.info("Starting DIC data population script...")
    logger.debug("Debugging mode enabled, detailed logs will be shown.")

    populate_dic_data()
