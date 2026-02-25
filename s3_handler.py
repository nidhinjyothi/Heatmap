import boto3
import cv2
from datetime import datetime
import config


def get_s3_client():
    """Creates and returns an S3 client using credentials from config."""
    return boto3.client(
        "s3",
        region_name=config.S3_CONFIG["region"],
        aws_access_key_id=config.S3_CONFIG["access_key_id"],
        aws_secret_access_key=config.S3_CONFIG["secret_access_key"],
    )


def upload_frame(frame, video_name, timestamp):
    """
    Encodes a frame as JPEG, uploads to S3, and returns the S3 path.

    Args:
        frame: OpenCV image (numpy array)
        video_name: Name of the video source
        timestamp: datetime object for the frame

    Returns:
        str: The S3 path (key) of the uploaded frame
    """
    s3 = get_s3_client()
    bucket = config.S3_CONFIG["bucket_name"]
    folder = config.S3_CONFIG["folder"]

    # Create a unique filename based on video name and timestamp
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{video_name}_{ts_str}.jpg"
    s3_key = f"{folder}/{filename}"

    # Encode frame to JPEG bytes
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    # Upload to S3
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )
    print(f"Frame uploaded to s3://{bucket}/{s3_key}")

    return s3_key
