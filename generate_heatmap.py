"""
Heatmap Generator — Fetches data from AWS RDS + S3 and generates a visual heatmap.

"""

import cv2
import numpy as np
import psycopg2
import boto3
import os
import config


def get_db_connection():
    """Connect to the AWS RDS database."""
    return psycopg2.connect(**config.DB_CONFIG)


def get_s3_client():
    """Create an S3 client."""
    return boto3.client(
        "s3",
        region_name=config.S3_CONFIG["region"],
        aws_access_key_id=config.S3_CONFIG["access_key_id"],
        aws_secret_access_key=config.S3_CONFIG["secret_access_key"],
    )


def download_frame_from_s3(frame_path):
    """Downloads a reference frame from S3 and returns it as an OpenCV image."""
    s3 = get_s3_client()
    bucket = config.S3_CONFIG["bucket_name"]

    response = s3.get_object(Bucket=bucket, Key=frame_path)
    image_bytes = response["Body"].read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_interval_report(video_name, start_time, end_time, store_id, output_path="heatmap_output.jpg"):
    """
    Generates a heatmap for a given time interval.

    Args:
        video_name: Name of the video (matches store config)
        start_time: Start timestamp string, e.g., "2026-02-23 15:00:00"
        end_time:   End timestamp string, e.g., "2026-02-23 15:30:00"
        store_id:   The ID of the store (fetched from config)
        output_path: Path to save the output heatmap image
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Fetch reference frame path from heatmap_evidences
    cur.execute(
        "SELECT frame_path FROM sapien_vision.heatmap_evidences WHERE video_name = %s ORDER BY id DESC LIMIT 1",
        (video_name,)
    )
    row = cur.fetchone()

    if not row:
        print(f"No reference frame found for video: {video_name}")
        conn.close()
        return

    # 2. Download reference frame from S3
    print(f"Downloading reference frame from S3: {row[0]}")
    bg_img = download_frame_from_s3(row[0])

    # 3. Fetch bounding box coordinates from heatmap_data
    cur.execute("""
        SELECT bounding_boxes FROM sapien_vision.heatmap_data
        WHERE store_id = %s AND recorded_at BETWEEN %s AND %s
        AND bounding_boxes IS NOT NULL
    """, (store_id, start_time, end_time))
    coord_rows = cur.fetchall()
    conn.close()

    if not coord_rows:
        print(f"No detection data found for Video: {video_name} (Store: {store_id}) in this interval.")
        return

    print(f"Found {len(coord_rows)} data points. Generating heatmap...")

    # 4. Build heatmap matrix
    h, w, _ = bg_img.shape
    h_matrix = np.zeros((h, w), dtype=np.float32)
    for r in coord_rows:
        if r[0]:  # bounding_boxes is not None
            for cx, cy in r[0]:
                if 0 <= cx < w and 0 <= cy < h:
                    h_matrix[cy, cx] += 1

    # 5. Visualize
    blur = cv2.GaussianBlur(h_matrix, (81, 81), 0)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    mask = (norm > 2).astype(np.uint8) * 255

    bg_clear = cv2.bitwise_and(bg_img, bg_img, mask=cv2.bitwise_not(mask))
    heat_only = cv2.bitwise_and(
        cv2.addWeighted(bg_img, 0.5, color, 0.5, 0),
        cv2.addWeighted(bg_img, 0.5, color, 0.5, 0),
        mask=mask
    )

    result = cv2.add(bg_clear, heat_only)

    # 6. Save output
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, output_path)
    
    cv2.imwrite(final_path, result)
    print(f"Heatmap saved to: {final_path}")

    # 7. Native Display (OpenCV Window)
    print("\nOpening display window... (Press any key to close)")
    cv2.namedWindow("Sapien Heatmap", cv2.WINDOW_NORMAL)
    cv2.imshow("Sapien Heatmap", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n--- Sapien Heatmap Generator ---")
    
    video_name = input("Enter Video Name (e.g., Main_Store_Cam_01): ").strip()
    
    # Lookup store_id from config.py based on video_name
    found_cam = None
    for cam in config.CAMERAS:
        if cam["name"] == video_name:
            found_cam = cam
            break
            
    if not found_cam:
        print(f"Error: Video name '{video_name}' not found in config.CAMERAS")
        # Optional: prompt for store_id if not found?
        store_id_input = input("Enter Store ID manually to continue: ").strip()
        if not store_id_input:
            exit(1)
        store_id = int(store_id_input)
    else:
        store_id = found_cam["store_id"]
        print(f"Using Store ID: {store_id} (found in config)")

    start_time = input("Enter Start Time (YYYY-MM-DD HH:MM:SS): ").strip()
    end_time   = input("Enter End Time   (YYYY-MM-DD HH:MM:SS): ").strip()
    
    output_path = input("Enter Output Filename [heatmap_output.jpg]: ").strip()
    if not output_path:
        output_path = "heatmap_output.jpg"

    get_interval_report(video_name, start_time, end_time, store_id, output_path)
