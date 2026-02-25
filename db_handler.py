import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import pytz
import config


def setup_database():
    """Connects to AWS RDS and ensures schema is ready."""
    print("Connecting to AWS RDS...")
    conn = psycopg2.connect(**config.DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # 1. Add bounding_boxes column to existing heatmap_data 
    cur.execute("""
        ALTER TABLE sapien_vision.heatmap_data
        ADD COLUMN IF NOT EXISTS bounding_boxes JSONB;
    """)
    print("heatmap_data: bounding_boxes column verified.")

    # 2. Create heatmap_evidences table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sapien_vision.heatmap_evidences (
            id SERIAL PRIMARY KEY,
            video_name VARCHAR(255),
            timestamp TIMESTAMP WITHOUT TIME ZONE,
            frame_path TEXT
        );
    """)
    print("heatmap_evidences: table verified.")

    print("AWS RDS connection established.")
    return conn, cur


def insert_batch(cur, conn, batch):
    """
    Inserts a batch of detection data into sapien_vision.heatmap_data.

    Each item in batch is a tuple:
        (store_id, region_id, recorded_at, value, bounding_boxes_json, created_at, updated_at)
    """
    query = """
        INSERT INTO sapien_vision.heatmap_data
            (store_id, region_id, recorded_at, value, bounding_boxes, created_at, updated_at)
        VALUES %s
    """
    execute_values(cur, query, batch)
    conn.commit()


def save_reference_frame(cur, conn, video_name, timestamp, frame_s3_path):
    """Saves the reference frame S3 path to heatmap_evidences."""
    cur.execute(
        """INSERT INTO sapien_vision.heatmap_evidences
           (video_name, timestamp, frame_path) VALUES (%s, %s, %s)""",
        (video_name, timestamp, frame_s3_path)
    )
    conn.commit()
