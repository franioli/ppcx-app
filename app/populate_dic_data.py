import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import django
import numpy as np
import pandas as pd
from django.utils import timezone
from PIL import Image

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

CAMERA_FOLDERS = {
    "Planpincieux_Tele": "Planpincieux_Tele",
    "Planpincieux_Wide": "Planpincieux_Wide",
}


def parse_day_dic_filename(
    filename: str | Path,
) -> tuple[datetime, datetime] | tuple[None, None]:
    parts = Path(filename).stem.split("_")
    if len(parts) < 3 or parts[0] != "day" or parts[1] != "dic":  # Fixed parsing logic
        logger.error(f"Filename does not match expected format: {filename}")
        return None, None

    cur_day_str, prev_day_str = parts[2].split("-")
    cur_day = datetime.strptime(cur_day_str, "%Y%m%d")
    prev_day = datetime.strptime(prev_day_str, "%Y%m%d")

    return cur_day, prev_day


def parse_image_filename(filename: Path | str) -> datetime | None:
    """Extract date information from filename.

    Args:
        filename: The filename string, expected format: PPCX_1_2024_08_23_17_00_13_REG.jpg

    Returns:
        A datetime object if parsing is successful, otherwise None.
    """
    filename = Path(filename)
    if not filename.name.startswith("PPCX_"):
        logger.error(f"Filename does not start with 'PPCX_' as expected: {filename}")
        return None

    parts = filename.name.split("_")
    if len(parts) >= 7:
        year = int(parts[2])
        month = int(parts[3])
        day = int(parts[4])
        hour = int(parts[5])
        minute = int(parts[6])
        return datetime(year, month, day, hour, minute)
    return None


def read_couples_file(couples_file: Path | str) -> list[tuple[str, str]]:
    """Get image couples from a couples file.

    Args:
        couples_file: Path to the couples file.

    Returns:
        List of tuples, each containing two image filenames.
    """
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
    dic_file: Path | str, image: Path | str = None
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
        dic_results = pd.read_csv(dic_file)
    except Exception as e:
        logger.error(f"Error reading DIC file: {e}")
        return None

    magnitudes = dic_results["V"].values
    max_magnitude = np.max(magnitudes) if len(magnitudes) > 0 else 1.0

    points = []
    vectors = []
    magnitudes_list = []
    for idx, row in dic_results.iterrows():
        x, y = int(row["X"]), -int(row["Y"])  # Note: y component is reversed

        if image is not None:
            img = Image.open(image)
            width, height = img.size
            if not (0 <= x < width and 0 <= y < height):
                continue
        else:
            logger.debug(
                f"Image not provided, skipping bounds check for point ({x}, {y})"
            )

        dx, dy = row["EW"], -row["NS"]  # Note: NS component is reversed
        magnitude = row["V"]

        points.append([x, y])
        vectors.append([dx, dy])
        magnitudes_list.append(magnitude)

    if not points:
        return None

    return {
        "points": np.array(points),
        "vectors": np.array(vectors),
        "magnitudes": np.array(magnitudes_list),
        "max_magnitude": max_magnitude,
    }


def populate_dic_data():
    logger.info("Starting DIC data population script...")

    dic_data_processed = 0
    dic_data_added = 0
    dic_data_failed = 0

    # Iterate through the years in the DIC data directory
    for year_dir in sorted(DIC_DATA_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        logger.info(f"Processing year: {year_dir.name}")

        for cam in sorted(CAMERA_FOLDERS.keys()):
            cam_dir = year_dir / cam
            if not cam_dir.is_dir():
                continue

            couples_dir = cam_dir / DIC_COUPLES_DIR
            if not couples_dir.is_dir():
                logger.error(f"No couples directory found for {cam} in {year_dir.name}")
                continue
            dic_res_dir = cam_dir / DIC_RESULTS_DIR
            if not dic_res_dir.is_dir():
                logger.error(f"No results directory found for {cam} in {year_dir.name}")
                continue

            dic_images_dir = cam_dir / DIC_IMAGES_DIR
            if not dic_images_dir.is_dir():
                logger.error(f"No images directory found for {cam} in {year_dir.name}")
                continue

            logger.info(f"Processing camera: {cam} in year {year_dir.name}")

            for daily_dic in sorted(dic_res_dir.iterdir()):
                if not daily_dic.is_file() or not daily_dic.name.startswith("day_dic_"):
                    continue

                try:
                    cur_day, prev_day = parse_day_dic_filename(daily_dic.name)
                    if cur_day is None or prev_day is None:
                        continue

                    # Build the couples and images file path
                    couples_file = (
                        couples_dir / f"couples_{cur_day.strftime('%Y%m%d')}.txt"
                    )
                    couples = read_couples_file(couples_file)

                    if not couples:
                        logger.warning(f"No couples found for {couples_file}")
                        continue

                    # Read DIC results
                    dic_results = read_dic_results(daily_dic)
                    if dic_results is None:
                        logger.warning(f"No DIC results found in {daily_dic}")
                        continue

                    # For the moment ignore the multiple couples, but add the DIC analysis only on a daily basis.
                    # Use the first couple for analysis
                    master_img, slave_img = couples[0]
                    logger.debug(
                        f"Processing couple: {master_img} -> {slave_img} from {daily_dic.name}"
                    )

                    try:
                        # Parse timestamps from image names
                        master_timestamp = parse_image_filename(master_img)
                        slave_timestamp = parse_image_filename(slave_img)
                        if not master_timestamp or not slave_timestamp:
                            logger.error(
                                f"Could not parse timestamps from {master_img}, {slave_img}"
                            )
                            continue
                        master_timestamp = timezone.make_aware(master_timestamp)
                        slave_timestamp = timezone.make_aware(slave_timestamp)

                        # Check if analysis already exists
                        existing_analysis = DICAnalysis.objects.filter(
                            master_image_path=master_img, slave_image_path=slave_img
                        ).first()

                        if existing_analysis:
                            logger.debug(
                                f"DIC analysis already exists for {master_img} -> {slave_img}"
                            )
                            continue

                        # Check if images exist in the database to reference them
                        master_img_path = dic_images_dir / master_img.replace(
                            "_REG", ""
                        )
                        slave_img_path = dic_images_dir / slave_img.replace("_REG", "")
                        master_image = ImageModel.objects.filter(
                            acquisition_timestamp=master_timestamp
                        ).first()

                        # Create DICAnalysis entry
                        dic_analysis = DICAnalysis(
                            analysis_timestamp=timezone.now(),  # Use current time for analysis timestamp
                            master_image_path=master_img,
                            slave_image_path=slave_img,
                            master_timestamp=master_timestamp,
                            slave_timestamp=slave_timestamp,
                            software_used="PyLamma",
                        )
                        dic_analysis.save()
                        dic_data_added += 1
                        logger.info(
                            f"Added DICAnalysis for {master_img} -> {slave_img}"
                        )

                        # Create DICResult entries
                        for point, vector, magnitude in zip(
                            dic_results["points"],
                            dic_results["vectors"],
                            dic_results["magnitudes"],
                            strict=False,
                        ):
                            dic_result_entry = DICResult(
                                analysis=dic_analysis,
                                seed_x_ref_px=int(point[0]),
                                seed_y_ref_px=int(point[1]),
                                target_x_sec_px=float(
                                    point[0] + vector[0]
                                ),  # Calculate target position
                                target_y_sec_px=float(point[1] + vector[1]),
                                displacement_x_px=float(vector[0]),
                                displacement_y_px=float(vector[1]),
                                correlation_score=float(magnitude),
                            )
                            dic_result_entry.save()
                            dic_data_processed += 1

                    except Exception as e:
                        logger.error(
                            f"Error processing couple {master_img}, {slave_img}: {e}"
                        )
                        dic_data_failed += 1

                except Exception as e:
                    logger.error(f"Error processing {daily_dic}: {e}")
                    dic_data_failed += 1

    logger.info(
        f"DIC data population script finished. Processed: {dic_data_processed}, Added: {dic_data_added}, Failed: {dic_data_failed}"
    )


if __name__ == "__main__":
    populate_dic_data()
