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

DIC_DATA_DIR = Path("/media/francesco/extreme/cnr/planpicieux/db_import")
DIC_COUPLES_DIR = "liste_coppie"
DIC_RESULTS_DIR = "matrici_spostamento"
DIC_IMAGES_DIR = "coregistrate"

DIC_RESULTS_PATTERN = "day_dic_*.txt"

CAMERA_FOLDERS = [
    # "Planpincieux_Tele",
    "Planpincieux_Wide",
]


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


def create_dic_analysis_for_pair(
    reference_date: datetime, master_img: str, slave_img: str, dic_results_dir: Path
) -> int:
    """Create DIC analysis for a specific image pair.

    Args:
        reference_date: Reference date for the analysis
        master_img: Master image filename
        slave_img: Slave image filename
        dic_results_dir: Directory containing DIC results files

    Returns:
        1: Successfully added
        0: Skipped (already exists or no data)
        -1: Failed (error during processing)
    """
    # Parse image timestamps
    master_timestamp = parse_image_filename(master_img)
    slave_timestamp = parse_image_filename(slave_img)

    if not master_timestamp or not slave_timestamp:
        logger.error(f"Failed to parse timestamps from {master_img} or {slave_img}")
        return -1

    # Make timestamps timezone-aware
    master_timestamp_tz = timezone.make_aware(master_timestamp)
    slave_timestamp_tz = timezone.make_aware(slave_timestamp)

    # Check if analysis already exists
    if DICAnalysis.objects.filter(
        master_timestamp=master_timestamp_tz,
        slave_timestamp=slave_timestamp_tz,
    ).exists():
        logger.debug(f"Analysis already exists for pair {master_img} -> {slave_img}")
        return 0  # Skipped - already exists

    # Find corresponding DIC results file
    dic_results_file = find_dic_results_for_pair(master_img, slave_img, dic_results_dir)
    if not dic_results_file:
        logger.warning(
            f"No DIC results file found for pair {master_img} -> {slave_img}"
        )
        return -1  # Failed - no results file

    try:
        # Use atomic transaction to ensure all-or-nothing database operations
        with transaction.atomic():
            # Read DIC results
            dic_results = read_dic_results(dic_results_file)
            if not dic_results:
                logger.warning(f"No valid DIC results in {dic_results_file}")
                return -1  # Failed - no valid results

            # Find related images in database
            master_name = Path(master_img).name.replace("_REG", "")
            slave_name = Path(slave_img).name.replace("_REG", "")
            master_image = ImageModel.objects.filter(
                file_path__contains=master_name
            ).first()
            slave_image = ImageModel.objects.filter(
                file_path__contains=slave_name
            ).first()

            # Create DIC analysis
            dic_analysis = DICAnalysis.objects.create(
                reference_date=reference_date,
                master_image=master_image,
                slave_image=slave_image,
                master_timestamp=master_timestamp_tz,
                slave_timestamp=slave_timestamp_tz,
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
                    f"Successfully added {len(dic_result_entries)} DIC results for pair "
                    f"{master_img} -> {slave_img} (date: {reference_date.strftime('%Y-%m-%d')})"
                )
                return 1  # Success
            else:
                # No results to create, rollback transaction
                transaction.set_rollback(True)
                return -1  # Failed - no results to create

    except Exception as e:
        logger.error(
            f"Error creating DIC analysis for pair {master_img} -> {slave_img}: {e}"
        )
        return -1  # Failed - error during processing


def process_couples_file(
    couples_file: Path, dic_results_dir: Path
) -> tuple[int, int, int]:
    """Process a single couples file and create DIC analyses for all pairs.

    Args:
        couples_file: Path to the couples file
        dic_results_dir: Directory containing DIC results files

    Returns:
        Tuple of (added, skipped, failed) counts
    """
    # Extract reference date from couples filename
    reference_date = parse_couples_filename(couples_file)
    if not reference_date:
        logger.error(f"Failed to parse reference date from {couples_file}")
        return 0, 0, 1

    # Read image pairs from couples file
    couples = read_couples_file(couples_file)
    if not couples:
        logger.warning(f"No couples found in {couples_file}")
        return 0, 0, 0

    logger.debug(
        f"Processing {len(couples)} pairs for date {reference_date.strftime('%Y-%m-%d')}"
    )

    # Process each image pair
    added = skipped = failed = 0
    for master_img, slave_img in couples:
        result = create_dic_analysis_for_pair(
            reference_date, master_img, slave_img, dic_results_dir
        )

        if result == 1:
            added += 1
        elif result == 0:
            skipped += 1
        elif result == -1:
            failed += 1

    return added, skipped, failed


def populate_dic_data():
    """Main function to populate DIC data by processing couples files."""
    logger.info("Starting DIC data population (couples-first approach)...")

    total_added = total_skipped = total_failed = 0
    total_couples_files = 0

    # First pass: count total couples files to process
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

    # Process each year/camera combination
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

                    # Show statistics in progress bar
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
    populate_dic_data()
