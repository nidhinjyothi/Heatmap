# Sapiens Heatmap Pipeline

A technical toolset for processing surveillance video to generate heatmaps.

## Directory Structure

### 1. GPU Scripts (Run in Google Colab)
- **`main.py`**: Standard processing script for L4 GPUs (Inference Batch Size 1).
- **`main_batch.py`**: High-performance script for A100 and L4 GPUs (Inference Batch Size 16).
- **`model_handler.py`**: Handles YOLOv8 model loading and TensorRT engine exports (.engine files).
- **`main_colab.ipynb`**: The notebook interface. Use the terminal cells to run the `.py` scripts.

### 2. Local Scripts (Run in VS Code Terminal)
- **`generate_heatmap.py`**: Fetches data from AWS and overlays a heatmap on the reference frame. Saves to `/outputs`.
- **`db_utils.py`**: Interactive CLI for database maintenance (deleting test data, checking counts).

### 3. Support Files
- **`config.py`**: Centralized configuration for intervals, database settings, and camera paths.
- **`db_handler.py`**: Logic for AWS RDS interactions (Batch writes).
- **`s3_handler.py`**: Logic for uploading reference frames to AWS S3.
- **`.env`**: (User Created) Stores secret keys. **Do not share this file.**
- **`.env.template`**: Reference for which variables to put in your `.env`.

---

## Setup & Execution

1. **Local Setup**:
   ```bash
   pip install -r requirements.txt
   cp .env.template .env
   # Fill .env with RDS and S3 keys
   ```

2. **Run Inference (Colab)**:
   - Use `main.py` for L4 or `main_batch.py` for A100 and L4 (Batched).
   - Videos should be listed in `config.py`.

3. **Generate Visuals (Local)**:
   ```bash
   python generate_heatmap.py
   ```

---

## Hardware Performance Logs

### A100 Batched (Batch Size 16)
```text
A100 Batched Throughput: 239.5 FPS (4.2ms avg per frame)

=======================================================
  STEP-BY-STEP TIMING REPORT (A100 BATCHED)
=======================================================
  video_seek             total=  0.02s  avg=  22.2ms  calls=1  eff_fps=45.1
  s3_upload              total=  0.83s  avg= 831.4ms  calls=1  eff_fps=1.2
  video_read             total=  2.11s  avg=  27.0ms  calls=78  eff_fps=37.0
  inference              total=  0.35s  avg=   4.4ms  calls=80  eff_fps=229.6
  batch_build            total=  0.03s  avg=   0.4ms  calls=78  eff_fps=2408.9
  db_queue               total=  0.00s  avg=   0.0ms  calls=1  eff_fps=32212.3
=======================================================
```

### L4 Batched (Batch Size 16)
```text
L4 Batched Throughput: 150.3 FPS (6.7ms avg per frame)

=======================================================
  STEP-BY-STEP TIMING REPORT (L4 BATCHED)
=======================================================
  video_seek             total=  0.02s  avg=  22.2ms  calls=1  eff_fps=45.0
  s3_upload              total=  0.72s  avg= 717.6ms  calls=1  eff_fps=1.4
  video_read             total=  2.11s  avg=  27.0ms  calls=78  eff_fps=37.0
  inference              total=  0.53s  avg=   6.7ms  calls=80  eff_fps=150.0
  batch_build            total=  0.03s  avg=   0.4ms  calls=78  eff_fps=2475.5
=======================================================
```

### L4 Serial (Baseline)
```text
True L4 inference FPS: 130.1 FPS (7.7ms per frame)
```

---

## 🛠 Maintenance
Use `python db_utils.py` to:
- Clear data for a specific video name.
- Remove old rows beyond a certain ID threshold to keep the database light.
- Check current row counts in RDS.
