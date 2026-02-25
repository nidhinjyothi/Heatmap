import cv2
import json
import os
import shutil
import time
import queue
import threading
import pytz
from datetime import datetime, timedelta
from tqdm import tqdm
import torch

import config
import db_handler
import model_handler
import s3_handler
import supervision as sv

# ── Timing Utilities ─────────────────────────────────────────────────────────

class StepTimer:
    """Accumulates time across multiple calls for each named step."""
    def __init__(self):
        self._totals = {}
        self._counts = {}
        self._start  = {}

    def start(self, name):
        self._start[name] = time.perf_counter()

    def stop(self, name, count=1):
        elapsed = time.perf_counter() - self._start[name]
        self._totals[name] = self._totals.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + count

    def report(self, total_wall_seconds):
        print("\n" + "=" * 55)
        print("  STEP-BY-STEP TIMING REPORT (BATCHED)")
        print("=" * 55)
        for name, total in self._totals.items():
            count = self._counts[name]
            avg_ms = (total / count) * 1000 if count > 0 else 0
            fps_equiv = count / total if total > 0 else 0
            print(f"  {name:<22} total={total:6.2f}s  avg={avg_ms:6.1f}ms  calls={count}  eff_fps={fps_equiv:.1f}")
        print(f"\n  Wall-clock total     : {total_wall_seconds:.2f}s")
        print("=" * 55 + "\n")

# ── DB Worker Thread ──────────────────────────────────────────────────────────

def db_worker(db_queue, conn, cur, timer):
    """Threaded worker to handle database inserts asynchronously."""
    while True:
        batch = db_queue.get()
        if batch is None:
            break
        db_handler.insert_batch(cur, conn, batch)
        db_queue.task_done()

# ── Batch Processing ─────────────────────────────────────────────────────────

def process_camera_batched(camera, model, conn, cur, db_queue, timer, batch_size=16):
    """Processes a single camera video using batched frame seeking."""
    video_name = camera["name"]
    store_id   = camera["store_id"]
    region_id  = camera["region_id"]

    print(f"\n{'─'*55}")
    print(f"  CAMERA  : {video_name}")
    print(f"  FILE    : {camera['video_path']}")
    print(f"{'─'*55}")

    local_path = f"/content/{os.path.basename(camera['video_path'])}"
    if not os.path.exists(local_path):
        print(f"  Copying to local disk: {local_path}")
        shutil.copy(camera["video_path"], local_path)
    video_path = local_path

    # 1. Open video & compute frame indices
    timer.start("video_seek")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * config.LOG_INTERVAL_SECONDS)
    frame_indices = list(range(0, total_frames, step))
    timer.stop("video_seek", count=1)

    print(f"  FPS={fps:.1f}  Frames={total_frames}")
    print(f"  Inferring on {len(frame_indices)} frames (Batch Size {batch_size})\n")

    IST = pytz.timezone('Asia/Kolkata')
    start_real_ts = datetime.now(IST).replace(tzinfo=None)
    data_batch_log = []

    # 2. Reference frame
    timer.start("s3_upload")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        s3_path = s3_handler.upload_frame(first_frame, video_name, start_real_ts)
        db_handler.save_reference_frame(cur, conn, video_name, start_real_ts, s3_path)
        print(f"  Reference frame → S3: {s3_path}")
    timer.stop("s3_upload", count=1)

    # 3. Batched Inference Loop
    for i in tqdm(range(0, len(frame_indices), batch_size), desc=f"  {video_name}"):
        current_batch_indices = frame_indices[i : i + batch_size]
        frames = []
        
        # Collect frames
        timer.start("video_read")
        for idx in current_batch_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            
        orig_count = len(frames)
        if 0 < orig_count < batch_size:
            last_frame = frames[-1]
            for _ in range(batch_size - orig_count):
                frames.append(last_frame)
        timer.stop("video_read", count=orig_count)

        if not frames: break

        # Run Batched Inference
        timer.start("inference")
        results = model(frames, conf=0.25, verbose=False)
        timer.stop("inference", count=batch_size) # We decode batch_size even if padded

        # Process results
        timer.start("batch_build")
        for idx_in_batch in range(orig_count):
            res = results[idx_in_batch]
            frame_idx = current_batch_indices[idx_in_batch]
            detections = sv.Detections.from_ultralytics(res)
            detections = detections[detections.class_id == 0]
            
            video_offset_s = frame_idx / fps
            recorded_at = start_real_ts + timedelta(seconds=video_offset_s)
            person_count = len(detections.xyxy)
            
            bboxes = []
            for box in detections.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                bboxes.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])

            data_batch_log.append((
                store_id, region_id, recorded_at, person_count,
                json.dumps(bboxes), start_real_ts, start_real_ts,
            ))
        timer.stop("batch_build", count=orig_count)

        if len(data_batch_log) >= config.DB_BATCH_SIZE:
            timer.start("db_queue")
            db_queue.put(list(data_batch_log))
            data_batch_log = []
            timer.stop("db_queue", count=1)

    if data_batch_log:
        timer.start("db_queue")
        db_queue.put(data_batch_log)
        timer.stop("db_queue", count=1)

    cap.release()

def main():
    wall_start = time.perf_counter()
    timer = StepTimer()
    conn, cur = db_handler.setup_database()
    db_queue = queue.Queue()
    worker_thread = threading.Thread(target=db_worker, args=(db_queue, conn, cur, timer), daemon=True)
    worker_thread.start()

    print("\nLoading TensorRT model...")
    model = model_handler.load_model_batch(config.MODEL_PATH, config.ENGINE_PATH, batch_size=16)

    # Batching Benchmark
    print("\nBenchmarking Batching (Batch Size 16)...")
    import numpy as np
    dummy_batch = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(16)]
    _ = model(dummy_batch, verbose=False) # warmup
    bm_start = time.perf_counter()
    for _ in range(5):
        model(dummy_batch, verbose=False)
    bm_elapsed = time.perf_counter() - bm_start
    batched_fps = (16 * 5) / bm_elapsed
    print(f"Batched Throughput: {batched_fps:.1f} FPS ({bm_elapsed/(16*5)*1000:.1f}ms avg per frame)\n")

    for camera in config.CAMERAS:
        process_camera_batched(camera, model, conn, cur, db_queue, timer, batch_size=16)

    print("\nWaiting for remaining DB writes...")
    db_queue.put(None)
    worker_thread.join()
    wall_elapsed = time.perf_counter() - wall_start
    timer.report(wall_elapsed)

if __name__ == "__main__":
    main()
