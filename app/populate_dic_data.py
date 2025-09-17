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

logger = logging.getLogger("ppcx")  # Use the logger from the ppcx_app module


# Paths of the directories containing DIC data and where to save HDF5 files on the host system
DIC_DATA_DIR = Path("/home/fioli/storage/fms/DIC_python")
DIC_H5_DIR_HOST = Path("/home/fioli/storage/francesco/ppcx_db/db_data")

# List of camera folders to process
CAMERA_FOLDERS = [
    "Planpincieux_Tele",
    "Planpincieux_Wide",
]

# Names of directories containing DIC data within each camera folder
DIC_COUPLES_DIR = "liste_coppie"
DIC_RESULTS_DIR = "matrici_spostamento"
DIC_IMAGES_DIR = "coregistrate"
DIC_RESULTS_PATTERN = "day_dic_*.txt"

# If True, master image is listed first in the day_dic_xxx.txt filenames
MASTER_IS_FIRST = True

# If True, master date is used as reference date for the DIC entry
MASTER_IS_REFERENCE = True

SOFTWARE_USED = "PyLamma"

# If True, existing DIC entries will be overwritten
OVERWRITE_EXISTING = False

# Path as seen inside the container (it MUST match the mount in docker-compose). DO NOT CHANGE THIS!
DIC_H5_DIR_CONTAINER = Path("/ppcx/data")

# ======= End of configuration =======


def parse_dic_result_filename(
    dic_filename: str | Path,
) -> tuple[datetime, datetime] | None:
    """
    Parse DIC result filename to extract slave (reference) and master dates.

    Expected format: day_dic_YYYYMMDD-YYYYMMDD.txt
    Returns (slave_date, master_date) as datetime objects.
    """
    try:
        dic_filename = Path(dic_filename)
        if not dic_filename.name.startswith("day_dic_"):
            return None
        date_part = dic_filename.stem.split("_")[2]  # 'YYYYMMDD-YYYYMMDD'
        slave_str, master_str = date_part.split("-")
        slave_date = datetime.strptime(slave_str, "%Y%m%d")
        master_date = datetime.strptime(master_str, "%Y%m%d")
        return slave_date, master_date
    except Exception as e:
        logger.error(f"Failed to parse DIC result filename: {dic_filename} ({e})")
        return None


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


def find_couples_file_for_date(
    couples_dir: Path, reference_date: datetime
) -> Path | None:
    """
    Find the couples file for a given reference date in the couples directory.
    """
    couples_file = couples_dir / f"couples_{reference_date.strftime('%Y%m%d')}.txt"
    if couples_file.exists():
        return couples_file
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


def save_dic_results_to_hdf5(dic_data: dict, file_path: str | Path) -> bool:
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

    Returns:
        bool: True if saved successfully, False otherwise
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(file_path, "w") as f:
            f.create_dataset("points", data=dic_data["points"], dtype="int32")
            f.create_dataset("vectors", data=dic_data["vectors"], dtype="float32")
            f.create_dataset("magnitudes", data=dic_data["magnitudes"], dtype="float32")
            f.create_dataset(
                "max_magnitude",
                data=np.array(dic_data["max_magnitude"], dtype="float32"),
            )
    except Exception as e:
        logger.error(f"Failed to save DIC results to HDF5: {e}")
        return False
    return True


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


def create_dic_from_result_file(
    dic_results_file: Path,
    couples_file: Path,
    reference_date: datetime,
    master_is_first: bool = True,
    take_middle_if_multiple: bool = True,
    overwrite: bool = False,
) -> int:
    """
    Create a DIC entry from a DIC result file and corresponding couples file.

    Args:
        dic_results_file (Path): Path to the DIC results file.
        couples_file (Path): Path to the couples file containing image pairs.
        reference_date (datetime): The reference date for the DIC entry.
        take_middle_if_multiple (bool): If True and multiple couples exist, take the middle pair.
        overwrite (bool): If True, overwrite existing DIC entries for the same image pair.

    Returns:
        int: 1 if DIC was created successfully, 0 if it already exists (skipped), -1 if there was an error or no valid data (failed).
    """
    couples = read_couples_file(couples_file)
    if not couples:
        logger.warning(f"No couples found in {couples_file}")
        return -1

    if MASTER_IS_FIRST:
        master_date, slave_date = parse_dic_result_filename(dic_results_file)
    else:
        slave_date, master_date = parse_dic_result_filename(dic_results_file)

    # Select only couples matching the dates in the DIC results filename
    filtered_couples = []
    for master_img, slave_img in couples:
        m_timestamp = parse_image_filename(master_img)
        s_timestamp = parse_image_filename(slave_img)
        if not m_timestamp or not s_timestamp:
            continue
        if (
            m_timestamp.date() == master_date.date()
            and s_timestamp.date() == slave_date.date()
        ) or (
            m_timestamp.date() == slave_date.date()
            and s_timestamp.date() == master_date.date()
        ):
            filtered_couples.append((master_img, slave_img))

    # Find the image pair corresponding to the DIC results file
    if not take_middle_if_multiple and len(filtered_couples) > 1:
        # Take the middle pair if multiple pairs exist
        middle_index = len(filtered_couples) // 2
        couple = filtered_couples[middle_index]
    else:
        # Take the last pair (most recent)
        couple = filtered_couples[-1]

    if master_is_first:
        master_img, slave_img = couple
    else:
        slave_img, master_img = couple

    master_timestamp = parse_image_filename(master_img)
    slave_timestamp = parse_image_filename(slave_img)
    if not master_timestamp or not slave_timestamp:
        logger.error(f"Failed to parse timestamps from {master_img} or {slave_img}")
        return -1
    master_timestamp_tz = timezone.make_aware(master_timestamp)
    slave_timestamp_tz = timezone.make_aware(slave_timestamp)

    # Check if DIC already exists, skip or overwrite based on flag
    dt_hours = int((slave_timestamp_tz - master_timestamp_tz).total_seconds() / 3600)
    cur_dic_qs = DIC.objects.filter(
        master_timestamp=master_timestamp_tz,
        slave_timestamp=slave_timestamp_tz,
        dt_hours=dt_hours,
    )
    if cur_dic_qs.exists() and not overwrite:
        logger.debug(
            f"DIC file {dic_results_file.name} already exists for pair {master_img} -> {slave_img}. Skipping."
        )
        return 0
    elif cur_dic_qs.exists() and overwrite:
        logger.debug(
            f"DIC file {dic_results_file.name} already exists for pair {master_img} -> {slave_img}. Overwriting."
        )
        cur_dic_qs.delete()

    with transaction.atomic():
        dic_data = read_dic_results(dic_results_file)
        if not dic_data:
            raise Exception("Invalid or empty DIC data.")

        # Find related images in database
        master_name = Path(master_img).name.replace("_REG", "")
        slave_name = Path(slave_img).name.replace("_REG", "")
        master_image = ImageModel.objects.filter(
            file_path__contains=master_name
        ).first()
        if not master_image:
            raise Exception(f"Master image {master_name} not found in database.")
        slave_image = ImageModel.objects.filter(file_path__contains=slave_name).first()
        if not slave_image:
            raise Exception(f"Slave image {slave_name} not found in database.")

        # Create DIC entry first to get the id
        dic = DIC.objects.create(
            reference_date=reference_date,
            master_image=master_image,
            slave_image=slave_image,
            master_timestamp=master_timestamp_tz,
            slave_timestamp=slave_timestamp_tz,
            result_file_path="",  # Temporary, will update after saving h5
            software_used=SOFTWARE_USED,
        )

        # Compose HDF5 file path (host) with DIC id
        h5_file_name = f"{dic.pk:08d}_{reference_date.strftime('%Y%m%d')}_{Path(master_img).stem}_{Path(slave_img).stem}.h5"
        save_dic_results_to_hdf5(dic_data, DIC_H5_DIR_HOST / h5_file_name)

        # Convert to container path for DB and store it
        h5_file_path_container = get_container_h5_path(DIC_H5_DIR_HOST / h5_file_name)
        dic.result_file_path = h5_file_path_container
        dic.save(update_fields=["result_file_path"])

        logger.debug(
            f"Successfully added DIC entry from {dic_results_file.name} ({master_img} -> {slave_img}) with ID {dic.pk}"
        )
        return 1


def main():
    logger.info("Starting DIC data population (results-first approach)...")
    total_added = total_skipped = total_failed = 0
    total_dic_files = 0
    year_cam_dicfiles = []

    for year_dir in sorted(DIC_DATA_DIR.iterdir(), reverse=True):
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
            dic_result_files = sorted(dic_results_dir.glob(DIC_RESULTS_PATTERN))
            year_cam_dicfiles.append(
                (year_dir, cam, dic_result_files, couples_dir, dic_results_dir)
            )
            total_dic_files += len(dic_result_files)
    logger.info(
        f"Found {total_dic_files} DIC result files to process across "
        f"{len(year_cam_dicfiles)} year/camera combinations"
    )

    for (
        year_dir,
        cam,
        dic_result_files,
        couples_dir,
        dic_results_dir,
    ) in year_cam_dicfiles:
        if not dic_result_files:
            continue
        logger.info(
            f"Processing {year_dir.name}/{cam} with {len(dic_result_files)} DIC result files"
        )
        with tqdm(dic_result_files, desc=f"{year_dir.name}/{cam}") as pbar:
            for dic_results_file in pbar:
                try:
                    parsed = parse_dic_result_filename(dic_results_file)
                    if not parsed:
                        logger.error(
                            f"Could not parse DIC result filename: {dic_results_file}"
                        )
                        total_failed += 1
                        continue

                    # Determine reference date based on MASTER_IS_FIRST
                    if MASTER_IS_FIRST:
                        master_date, slave_date = parsed
                    else:
                        slave_date, master_date = parsed
                    reference_date = master_date if MASTER_IS_REFERENCE else slave_date

                    # Get couples used for DIC
                    couples_file = find_couples_file_for_date(
                        couples_dir, reference_date
                    )
                    if not couples_file:
                        logger.error(
                            f"Couples file for date {reference_date.strftime('%Y%m%d')} not found in {couples_dir}"
                        )
                        total_failed += 1
                        continue
                    result = create_dic_from_result_file(
                        dic_results_file,
                        couples_file,
                        reference_date,
                        master_is_first=MASTER_IS_FIRST,
                        take_middle_if_multiple=True,
                        overwrite=OVERWRITE_EXISTING,
                    )
                    if result == 1:
                        total_added += 1
                    elif result == 0:
                        total_skipped += 1
                    elif result == -1:
                        total_failed += 1
                    pbar.set_postfix(
                        added=total_added,
                        skipped=total_skipped,
                        failed=total_failed,
                    )
                except Exception as e:
                    total_failed += 1
                    logger.error(
                        f"Error processing {dic_results_file}: {e}", exc_info=True
                    )
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
    main()
