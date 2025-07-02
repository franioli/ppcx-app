import io
import os
from datetime import datetime

import pandas as pd
import requests
from PIL import Image

DB_HOST = os.environ.get("DB_HOST", "150.145.51.193")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "planpincieux")
DB_USER = os.environ.get("DB_USER", "")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

APP_HOST = os.environ.get("APP_HOST", "150.145.51.193")
APP_PORT = os.environ.get("APP_PORT", "8001")
GET_IMAGE_VIEW = os.environ.get("GET_IMAGE_VIEW", "images")


def get_dic_analysis_ids(
    db_engine,
    *,
    reference_date: str | datetime | None = None,
    master_timestamp: str | datetime | None = None,
    camera_id: int | None = None,
    camera_name: str | None = None,
) -> pd.DataFrame:
    """
    Get DIC analysis metadata (including IDs) for a specific date/timestamp and optional camera filter.
    """
    query = """
    SELECT 
        DIC.id as dic_id,
        CAM.camera_name,
        DIC.master_timestamp,
        DIC.slave_timestamp,
        DIC.master_image_id,
        DIC.slave_image_id,
        DIC.time_difference_hours
    FROM ppcx_app_dic DIC
    JOIN ppcx_app_image IMG ON DIC.master_image_id = IMG.id
    JOIN ppcx_app_camera CAM ON IMG.camera_id = CAM.id
    WHERE 1=1
    """
    params = []
    if reference_date is not None:
        query += " AND DATE(DIC.reference_date) = %s"
        params.append(str(reference_date))
    if master_timestamp is not None:
        query += " AND DATE(DIC.master_timestamp) = %s"
        params.append(str(master_timestamp))
    if camera_id is not None:
        query += " AND CAM.id = %s"
        params.append(camera_id)
    if camera_name is not None:
        query += " AND CAM.camera_name = %s"
        params.append(camera_name)
    query += " ORDER BY DIC.master_timestamp"
    return pd.read_sql(query, db_engine, params=tuple(params))


def get_dic_data(
    dic_id: int,
    app_host: str = APP_HOST,
    app_port: str = APP_PORT,
    filter_outliers: bool = False,
    tails_percentile: float = 0.01,
    min_velocity: float = -1,
) -> pd.DataFrame:
    """
    Fetch DIC displacement data from the Django API endpoint as a DataFrame.
    """
    url = f"http://{app_host}:{app_port}/API/dic/{dic_id}/"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch DIC data for id {dic_id}: {response.text}")

    data = response.json()
    # If the data is empty, raise an error
    if not data or "points" not in data:
        raise ValueError(f"No valid DIC data found for id {dic_id}")

    points = data["points"]
    vectors = data["vectors"]
    magnitudes = data["magnitudes"]

    # Convert to DataFrame
    df = pd.DataFrame(points, columns=["x", "y"])
    df["u"] = [v[0] for v in vectors]
    df["v"] = [v[1] for v in vectors]
    df["V"] = magnitudes

    # Filter out extreme tails based on the specified percentile
    if filter_outliers:
        prob_threshold = (tails_percentile, 1 - tails_percentile)
        velocity_percentiles = df["V"].quantile(prob_threshold).values
        df = df[
            (df["V"] >= velocity_percentiles[0]) & (df["V"] <= velocity_percentiles[1])
        ].reset_index(drop=True)

    # Filter out low velocity vectors if specified
    if min_velocity >= 0:
        df = df[df["V"] >= min_velocity].reset_index(drop=True)

    return df


def get_image(
    image_id: int,
    app_host: str = APP_HOST,
    app_port: str = APP_PORT,
    camera_name: str | None = None,
) -> Image.Image:
    """Get an image from the database by its ID and rotate if from Tele camera."""
    url = f"http://{app_host}:{app_port}/API/images/{image_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        # Rotate if camera_name is Tele (portrait mode)
        if camera_name is not None and "tele" in camera_name.lower():
            img = img.rotate(90, expand=True)  # 90° clockwise
        return img
    else:
        raise ValueError(f"Image with ID {image_id} not found.")
