import os
import django
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage, ExifTags

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

from glacier_monitoring_app.models import Camera, Image
from add_cam_image_data import add_image

CAMERA_DIR = Path("/data/Dati/HiRes/Tele")
IMAGE_EXTENSIONS = (".tif")

MONTH_NAME_TO_NUMBER = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
    "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}
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
    print(f"Warning: Could not parse month: '{month_str}'")
    return None

def extract_exif_data(image_path):
    try:
        with PILImage.open(image_path) as img:
            exif = {ExifTags.TAGS.get(k, k): v for k, v in img.getexif().items()}
        model = exif.get("Model", None)
        lens = exif.get("LensModel", None)
        # FocalLength often stored as a tuple in EXIF, e.g. (50, 1) -> 50.0
        fl = exif.get("FocalLength", None)
        if fl and isinstance(fl, tuple) and fl[1] != 0:
            fl = float(fl[0]) / float(fl[1])
        return (str(model) if model else None,
                str(lens) if lens else None,
                fl if fl else None)
    except Exception:
        return (None, None, None)

def populate_images():
    print("Starting image population script...")


    for year_item in CAMERA_DIR.iterdir():
        if not year_item.is_dir():
            continue
        try:
            year = int(year_item.name)
            if not (2000 < year < 2100):
                continue
        except ValueError:
            continue

        for month_item in year_item.iterdir():
            if not month_item.is_dir():
                continue
            month = parse_month(month_item.name)
            if not month:
                continue

            for image_file in month_item.iterdir():
                if not image_file.is_file() or image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Infer EXIF data
                exif_model, exif_lens, exif_focal = extract_exif_data(image_file)

                # If no camera is found yet, create one if user agrees
                if camera_obj is None:
                    user_resp = input(f"No camera '{db_camera_name}' found. Create a new one? [y/N] ")
                    if user_resp.lower() in ["y", "yes"]:
                        camera_name = f"{exif_model or 'Unknown'}-{exif_lens or 'Unknown'}-{int(exif_focal or 0)}"
                        camera_obj = Camera.objects.create(camera_name=camera_name)
                        print(f"Created camera: {camera_name}")
                    else:
                        print("Skipping image since camera does not exist.")
                        continue

                # Build acquisition timestamp (approx)
                try:
                    acquisition_timestamp = datetime(year, month, 1, 0, 0, 0)
                except ValueError as e:
                    print(f"Error building timestamp for {image_file}: {e}")
                    continue

                # Create the DB entry
                try:
                    image_instance = add_image(
                        camera=camera_obj,
                        acquisition_timestamp=acquisition_timestamp,
                        file_path=str(image_file),
                    )
                    # Optionally update the camera fields with exif if missing
                    if not camera_obj.model and exif_model:
                        camera_obj.model = exif_model
                        camera_obj.lens = exif_lens or ""
                        if exif_focal:
                            camera_obj.focal_length_mm = exif_focal
                        camera_obj.save()
                    print(f"Added image: {image_file} (ID: {image_instance.id})")
                except Exception as err:
                    print(f"Failed adding {image_file}: {err}")

    print("\nImage population script finished.")

if __name__ == "__main__":
    populate_images()