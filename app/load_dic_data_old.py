from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class DICdata:
    """Dataclass to store DIC results and associated metadata."""

    time: datetime
    time_str: str
    img: Image.Image
    img_path: Path
    points: np.ndarray
    vectors: np.ndarray
    magnitudes: np.ndarray
    max_magnitude: float


def load_data(
    base_dir: str | Path,
    day_date: datetime,
    hour: int | None = None,
    return_all: bool = False,
) -> None | DICdata | list[DICdata]:
    """Load all DICdata for a specific day, optionally filtering by hour.

    Args:
        base_dir: Base directory containing images and results.
        day_date: The day as a datetime object.
        hour: If specified, only return results for this hour.
        return_all: If True, return all results for the day as a list.

    Returns:
        A single DICdata object if hour is specified and found, a list of DICdata if return_all is True,
        or None if no data found.
    """
    base_dir = Path(base_dir)
    day_str = day_date.strftime("%Y%m%d")
    prev_day = day_date - timedelta(days=1)
    prev_day_str = prev_day.strftime("%Y%m%d")

    couples_file = base_dir / f"couples_{day_str}.txt"
    dic_file = base_dir / f"day_dic_{day_str}-{prev_day_str}.txt"

    couples = get_couples_for_day(couples_file)
    if not couples:
        print(f"No couples found for day {day_str}")
        return None

    dicdata_list: list[DICdata] = []

    for i, (img1_name, img2_name) in enumerate(couples):
        img2_path = base_dir / img2_name

        if not img2_path.exists():
            print(f"Image not found: {img2_path}")
            continue

        try:
            img = Image.open(img2_path)
        except Exception as e:
            print(f"Failed to load image: {img2_path} - {e}")
            continue

        img_time = parse_date_from_filename(img2_name)
        time_str = img_time.strftime("%Y-%m-%d %H:%M") if img_time else f"Image {i}"

        processed_data = read_dic_results(dic_file, img)
        if processed_data and img_time:
            dicdata = DICdata(
                time=img_time,
                time_str=time_str,
                img=img,
                img_path=img2_path,
                points=processed_data["points"],
                vectors=processed_data["vectors"],
                magnitudes=processed_data["magnitudes"],
                max_magnitude=processed_data["max_magnitude"],
            )
            dicdata_list.append(dicdata)

    if not dicdata_list:
        return None

    if hour is not None:
        # Return the DICdata for the specified hour (closest match)
        for d in dicdata_list:
            if d.time.hour == hour:
                return d
        print(f"No DIC result found for hour {hour} on {day_str}")
        return None

    if return_all:
        return dicdata_list

    # Default: return the first result
    return dicdata_list[0]


if __name__ == "__main__":
    data_dir = Path("ppcx_1")
    day_to_process = datetime(2024, 8, 23)
    dicdata = load_data(data_dir, day_to_process, hour=16)
