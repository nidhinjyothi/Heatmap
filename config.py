import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- PIPELINE CONFIGURATION ---

# Processing Settings
LOG_INTERVAL_SECONDS = 5    # Run inference every N seconds of video
DB_BATCH_SIZE = 200         # Rows to batch before writing to AWS RDS

# Model Settings
MODEL_PATH = "yolov8l.pt"
ENGINE_PATH = "yolov8l.engine"

# --- CAMERA MANIFEST ---
# Add one entry per camera. Each entry is processed sequentially.
CAMERAS = [
    {
        "video_path": "2.mp4",
        "name":       "Main_Store_Cam_01",
        "store_id":   1,
        "region_id":  1,
    },
    # Add more cameras here, e.g.:
    # {
    #     "video_path": "cam2_24hr.mp4",
    #     "name":       "Main_Store_Cam_02",
    #     "store_id":   1,
    #     "region_id":  2,
    # },
]

# --- AWS RDS (PostgreSQL) ---
DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),  
}

# --- AWS S3 ---
S3_CONFIG = {
    "bucket_name":       os.getenv("S3_BUCKET_NAME"),
    "folder":            os.getenv("S3_FOLDER", "heatmap_images"),
    "region":            os.getenv("S3_REGION", "ap-south-1"),
    "access_key_id":     os.getenv("S3_ACCESS_KEY_ID"),    
    "secret_access_key": os.getenv("S3_SECRET_ACCESS_KEY"),   
}
