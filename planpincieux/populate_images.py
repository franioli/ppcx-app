import os
import django
from datetime import datetime

# Configure Django settings
# Ensure this points to your project's settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planpincieux.settings")
django.setup()

# Import models and the add_image function
from glacier_monitoring_app.models import Camera # Assuming your models are here
# Ensure add_data.py is in the same directory or PYTHONPATH
from add_data import add_image

# --- Configuration ---
BASE_DATA_DIR = "/data/"  # Root directory of your image data

CAMERA_FOLDERS_AND_DB_NAMES = {
    "Wide": "PPCX_1_Wide",
    "Tele": "PPCX_2_Tele"
}
# Common image file extensions to look for
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

# Month name to number mapping (add more if needed)
MONTH_NAME_TO_NUMBER = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    # Italian (based on your example)
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
    "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}
# --- End Configuration ---

def parse_month(month_str):
    """
    Parses a month string (name or number) into a month integer.
    """
    month_str_lower = month_str.lower()
    if month_str_lower in MONTH_NAME_TO_NUMBER:
        return MONTH_NAME_TO_NUMBER[month_str_lower]
    try:
        month_num = int(month_str)
        if 1 <= month_num <= 12:
            return month_num
    except ValueError:
        pass
    print(f"Warning: Could not parse month: '{month_str}'")
    return None

def populate_images():
    """
    Main function to find images and add them to the database.
    """
    print("Starting image population script...")

    # Fetch camera objects from the database
    camera_objects = {}
    for folder_name, db_camera_name in CAMERA_FOLDERS_AND_DB_NAMES.items():
        try:
            camera_objects[folder_name] = Camera.objects.get(camera_name=db_camera_name)
            print(f"Successfully fetched camera: '{db_camera_name}'")
        except Camera.DoesNotExist:
            print(f"ERROR: Camera '{db_camera_name}' not found in the database. "
                  f"Please ensure it exists before running this script.")
            # If you want to create cameras if they don't exist, you can use add_camera here:
            # from planpincieux.add_data import add_camera
            # camera_objects[folder_name] = add_camera(camera_name=db_camera_name, easting=0, northing=0) # Add relevant details
            # print(f"Created placeholder camera: {db_camera_name}")
            return # Exit if a camera is crucial and not found/created

    hires_base_path = os.path.join(BASE_DATA_DIR, "HiRes")
    if not os.path.isdir(hires_base_path):
        print(f"ERROR: HiRes directory not found at '{hires_base_path}'")
        return

    for camera_folder_name, db_camera_name in CAMERA_FOLDERS_AND_DB_NAMES.items():
        current_camera_obj = camera_objects.get(camera_folder_name)
        if not current_camera_obj:
            print(f"Skipping {camera_folder_name} as camera object was not loaded.")
            continue

        path_to_camera_type_folder = os.path.join(hires_base_path, camera_folder_name)
        print(f"\nProcessing camera type: {camera_folder_name} (Database Name: {db_camera_name})")
        print(f"Looking in: {path_to_camera_type_folder}")

        if not os.path.isdir(path_to_camera_type_folder):
            print(f"Warning: Directory not found: '{path_to_camera_type_folder}'")
            continue

        for year_folder_name in os.listdir(path_to_camera_type_folder):
            path_to_year_folder = os.path.join(path_to_camera_type_folder, year_folder_name)
            if not os.path.isdir(path_to_year_folder):
                continue

            try:
                year = int(year_folder_name)
                # Basic validation for year folder
                if not (2000 < year < 2100): # Adjust range if necessary
                    print(f"  Skipping non-standard year folder: '{year_folder_name}'")
                    continue
            except ValueError:
                print(f"  Skipping non-numeric year folder: '{year_folder_name}'")
                continue
            
            print(f"  Processing Year: {year}")

            for month_folder_name in os.listdir(path_to_year_folder):
                path_to_month_folder = os.path.join(path_to_year_folder, month_folder_name)
                if not os.path.isdir(path_to_month_folder):
                    continue

                month = parse_month(month_folder_name)
                if month is None:
                    print(f"    Skipping month folder with unparsed name: '{month_folder_name}'")
                    continue
                
                print(f"    Processing Month: {month_folder_name} (Parsed as {month})")

                for image_filename in os.listdir(path_to_month_folder):
                    if image_filename.lower().endswith(IMAGE_EXTENSIONS):
                        full_image_path = os.path.join(path_to_month_folder, image_filename)
                        
                        # Construct acquisition_timestamp. Defaults to the 1st of the month at 00:00:00.
                        # If filenames contain more precise info, this part would need enhancement.
                        try:
                            acquisition_timestamp = datetime(year, month, 1, 0, 0, 0)
                        except ValueError as e:
                            print(f"      Error creating datetime for {year}-{month}-01 for image {image_filename}: {e}")
                            continue
                        
                        # Add image to database
                        try:
                            # Optional: Check if image already exists to prevent duplicates
                            # from glacier_monitoring_app.models import Image
                            # if Image.objects.filter(camera=current_camera_obj, file_path=full_image_path).exists():
                            #     print(f"      Image already exists: {full_image_path}")
                            #     continue
                            
                            image_instance = add_image(
                                camera=current_camera_obj,
                                acquisition_timestamp=acquisition_timestamp,
                                file_path=full_image_path
                                # Add any other relevant kwargs for add_image if needed
                            )
                            print(f"      Added: {full_image_path} (ID: {image_instance.id})")
                        except Exception as e:
                            print(f"      ERROR adding image {full_image_path} to database: {e}")
                            
    print("\nImage population script finished.")

if __name__ == "__main__":
    populate_images()