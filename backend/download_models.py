"""
Run this once after cloning to download all required models.
    python download_models.py
"""

print("\n── DeepShield Model Downloader ──────────────────────\n")

# Deepfake detector
try:
    from transformers import pipeline
    print("Downloading umm-maybe/AI-image-detector (~350MB)...")
    pipeline("image-classification", model="umm-maybe/AI-image-detector")
    print("✓ Deepfake detector ready\n")
except Exception as e:
    print(f"✗ Failed: {e}\n")

# YOLO
try:
    from ultralytics import YOLO
    print("Downloading YOLOv8n (~6MB)...")
    YOLO("yolov8n.pt")
    print("✓ YOLO ready\n")
except Exception as e:
    print(f"✗ YOLO failed (pip install ultralytics): {e}\n")

# Whisper
try:
    import whisper
    print("Downloading Whisper tiny (~150MB)...")
    whisper.load_model("tiny")
    print("✓ Whisper ready\n")
except Exception as e:
    print(f"✗ Whisper failed (pip install openai-whisper): {e}\n")

print("── All done. Run: uvicorn main:app --reload --port 8000 ──\n")
