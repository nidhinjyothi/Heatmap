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

    def stop(self, name):
        elapsed = time.perf_counter() - self._start[name]
        self._totals[name] = self._totals.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def report(self, total_wall_seconds):
        print("\n" + "=" * 55)
        print("  STEP-BY-STEP TIMING REPORT")
        print("=" * 55)
        for name, total in self._totals.items():
            count = self._counts[name]
            avg_ms = (total / count) * 1000
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
        t0 = time.perf_counter()
        db_handler.insert_batch(cur, conn, batch)
        elapsed = time.perf_counter() - t0
        # Accumulate DB time from worker thread (approximate, not in main timer)
        db_queue.task_done()


# ── Per-Camera Processing ─────────────────────────────────────────────────────

def process_camera(camera, model, conn, cur, db_queue, timer):
    """Processes a single camera video using direct frame seeking."""
    video_name = camera["name"]
    store_id   = camera["store_id"]
    region_id  = camera["region_id"]

    print(f"\n{'─'*55}")
    print(f"  CAMERA  : {video_name}")
    print(f"  FILE    : {camera['video_path']}")
    print(f"{'─'*55}")

    # ── 0. Copy video to local Colab disk to avoid Drive network latency ───
    local_path = f"/content/{os.path.basename(camera['video_path'])}"
    if not os.path.exists(local_path):
        print(f"  Copying to local disk: {local_path}")
        t_copy = time.perf_counter()
        shutil.copy(camera["video_path"], local_path)
        print(f"  Copy done in {time.perf_counter() - t_copy:.1f}s")
    else:
        print(f"  Using cached local copy: {local_path}")
    video_path = local_path

    # ── 1. Open video & compute frame indices to seek ──────────────────────
    timer.start("video_seek")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps
    step         = int(fps * config.LOG_INTERVAL_SECONDS)
    frame_indices = list(range(0, total_frames, step))
    timer.stop("video_seek")

    print(f"  FPS={fps:.1f}  Frames={total_frames}  Duration={duration_s:.1f}s")
    print(f"  Inferring on {len(frame_indices)} frames "
          f"(every {config.LOG_INTERVAL_SECONDS}s interval)\n")

    IST            = pytz.timezone('Asia/Kolkata')
    start_real_ts  = datetime.now(IST).replace(tzinfo=None)
    created_at     = start_real_ts
    data_batch_log = []

    # ── 2. Reference frame ─────────────────────────────────────────────────
    timer.start("s3_upload")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        s3_path = s3_handler.upload_frame(first_frame, video_name, start_real_ts)
        db_handler.save_reference_frame(cur, conn, video_name, start_real_ts, s3_path)
        print(f"  Reference frame → S3: {s3_path}")
    timer.stop("s3_upload")

    # ── 3. Main inference loop with frame seeking ──────────────────────────
    for frame_idx in tqdm(frame_indices, desc=f"  {video_name}"):

        # Seek directly to the required frame (O(1), no decode in between)
        timer.start("video_read")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        timer.stop("video_read")

        if not ret:
            break

        # Run detection model
        timer.start("inference")
        results    = model_handler.run_inference(model, frame)
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]   # person only
        timer.stop("inference")

        # Build DB row
        timer.start("batch_build")
        video_offset_s = frame_idx / fps
        recorded_at    = start_real_ts + timedelta(seconds=video_offset_s)
        person_count   = len(detections.xyxy)

        bboxes = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = box.astype(int)
            bboxes.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])

        data_batch_log.append((
            store_id, region_id, recorded_at, person_count,
            json.dumps(bboxes), created_at, created_at,
        ))
        timer.stop("batch_build")

        # Flush batch to DB worker
        if len(data_batch_log) >= config.DB_BATCH_SIZE:
            timer.start("db_queue")
            db_queue.put(data_batch_log)
            data_batch_log = []
            timer.stop("db_queue")

    # Final flush
    if data_batch_log:
        timer.start("db_queue")
        db_queue.put(data_batch_log)
        timer.stop("db_queue")

    cap.release()
    print(f"  Done: {len(frame_indices)} inferences recorded.")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    wall_start = time.perf_counter()
    timer = StepTimer()

    # 1. Connect to AWS RDS
    conn, cur = db_handler.setup_database()

    # 2. Start async DB writer thread
    db_queue = queue.Queue()
    worker_thread = threading.Thread(
        target=db_worker, args=(db_queue, conn, cur, timer), daemon=True
    )
    worker_thread.start()

    # 3. Load TensorRT model once (shared across all cameras)
    print("\nLoading TensorRT model...")
    t0 = time.perf_counter()
    model = model_handler.load_model(config.MODEL_PATH, config.ENGINE_PATH)
    print(f"Model ready in {time.perf_counter() - t0:.1f}s")

    # 4. GPU inference speed benchmark (10 random frames)
    print("\nBenchmarking true GPU inference speed...")
    import numpy as np
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _ = model_handler.run_inference(model, dummy)           # warmup
    bm_start = time.perf_counter()
    for _ in range(10):
        model_handler.run_inference(model, dummy)
    bm_elapsed = time.perf_counter() - bm_start
    true_fps = 10 / bm_elapsed
    print(f"True L4 inference FPS: {true_fps:.1f} FPS ({bm_elapsed/10*1000:.1f}ms per frame)\n")

    # 5. Process each camera sequentially
    for camera in config.CAMERAS:
        process_camera(camera, model, conn, cur, db_queue, timer)

    # 6. Wait for DB writes to complete
    print("\nWaiting for remaining DB writes...")
    db_queue.put(None)
    worker_thread.join()

    wall_elapsed = time.perf_counter() - wall_start
    timer.report(wall_elapsed)


if __name__ == "__main__":
    main()
