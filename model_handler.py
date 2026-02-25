import os
from ultralytics import YOLO

def load_model(model_path, engine_path):
    """Loads a YOLO model for standard serial processing (L4 default)."""
    print(f"Loading standard model: {model_path}")
    model = YOLO(model_path)
    
    if not os.path.exists(engine_path):
        print("Exporting to standard TensorRT engine for GPU...")
        model.export(format="engine", device=0, half=True, workspace=4)
    
    print(f"Loading specialized engine: {engine_path}")
    return YOLO(engine_path)

def load_model_batch(model_path, engine_path, batch_size=16):
    """Loads a YOLO model specialized for high-throughput batching (A100)."""
    print(f"Loading batched model: {model_path}")
    model = YOLO(model_path)
    
    if not os.path.exists(engine_path):
        print(f"Exporting to batched TensorRT engine (Batch={batch_size}) for A100 throughput...")
        model.export(format="engine", device=0, half=True, workspace=4, batch=batch_size)
    
    print(f"Loading specialized batched engine: {engine_path}")
    return YOLO(engine_path)

def run_inference(model, frame):
    """Runs inference on a single frame."""
    results = model(frame, conf=0.25, verbose=False)[0]
    return results
