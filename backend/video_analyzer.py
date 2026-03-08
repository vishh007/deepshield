"""
DeepShield v3 — Multi-Modal Video Analyzer
============================================
Handles all computer vision tasks SEPARATE from deepfake scoring:
  • Face detection + tracking     (MediaPipe or OpenCV DNN)
  • Object detection              (YOLOv8 nano — ultralytics)
  • Scene classification          (inferred from YOLO objects)
  • On-screen text detection      (EasyOCR)
  • Speech-to-text                (OpenAI Whisper tiny)

Each module degrades gracefully — if a library isn't installed,
that feature is skipped and the rest still work.
"""

import logging
import os
from typing import Optional

log = logging.getLogger("deepshield.analyzer")

# ── Module availability flags ─────────────────────────────
_YOLO        = None   # ultralytics YOLO model
_FACE_DET    = None   # mediapipe face detection
_OCR         = None   # easyocr reader
_WHISPER     = None   # whisper model

CAPABILITIES = {
    "face":   False,
    "object": False,
    "ocr":    False,
    "speech": False,
}


def load_analyzers():
    """Load all available analysis modules. Called once at startup."""
    global _YOLO, _FACE_DET, _OCR, _WHISPER

    # ── Face detection (MediaPipe) ────────────────────────
    try:
        import mediapipe as mp
        _mp_face = mp.solutions.face_detection
        _FACE_DET = _mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        CAPABILITIES["face"] = True
        log.info("[Analyzer] ✓ Face detection loaded (MediaPipe)")
    except ImportError:
        try:
            import cv2
            # Fallback: OpenCV Haar cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _FACE_DET = cv2.CascadeClassifier(cascade_path)
            CAPABILITIES["face"] = True
            log.info("[Analyzer] ✓ Face detection loaded (OpenCV Haar)")
        except Exception as e:
            log.warning(f"[Analyzer] ✗ Face detection unavailable: {e}")

    # ── Object detection (YOLOv8 nano) ───────────────────
    try:
        from ultralytics import YOLO
        _YOLO = YOLO("yolov8n.pt")   # downloads ~6MB on first run
        _YOLO.overrides["verbose"] = False
        CAPABILITIES["object"] = True
        log.info("[Analyzer] ✓ Object detection loaded (YOLOv8n)")
    except ImportError:
        log.warning("[Analyzer] ✗ Object detection unavailable (pip install ultralytics)")
    except Exception as e:
        log.warning(f"[Analyzer] ✗ Object detection failed: {e}")

    # ── OCR text detection (EasyOCR) ─────────────────────
    try:
        import easyocr
        _OCR = easyocr.Reader(["en"], gpu=False, verbose=False)
        CAPABILITIES["ocr"] = True
        log.info("[Analyzer] ✓ OCR loaded (EasyOCR)")
    except ImportError:
        log.warning("[Analyzer] ✗ OCR unavailable (pip install easyocr)")
    except Exception as e:
        log.warning(f"[Analyzer] ✗ OCR failed: {e}")

    # ── Speech recognition (Whisper) ─────────────────────
    try:
        import whisper
        _WHISPER = whisper.load_model("tiny")
        CAPABILITIES["speech"] = True
        log.info("[Analyzer] ✓ Speech recognition loaded (Whisper tiny)")
    except ImportError:
        log.warning("[Analyzer] ✗ Speech unavailable (pip install openai-whisper)")
    except Exception as e:
        log.warning(f"[Analyzer] ✗ Whisper failed: {e}")

    log.info(f"[Analyzer] Capabilities: {CAPABILITIES}")


# ── SCENE RULES from YOLO labels ──────────────────────────
_SCENE_RULES = [
    ({"tv", "laptop", "keyboard", "mouse", "monitor", "desk"},                  "Office / Workstation",  True),
    ({"car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "stop sign"}, "Street / Traffic",  False),
    ({"couch", "chair", "dining table", "bed", "refrigerator", "toilet"},       "Indoor / Home",         True),
    ({"tree", "sky", "grass", "mountain", "river", "ocean"},                    "Outdoor / Nature",      False),
    ({"microphone", "tie", "suit"},                                              "Interview / Broadcast", True),
    ({"person"},                                                                 "General / People",      None),
    ({"bottle", "wine glass", "cup", "fork", "knife"},                          "Dining / Social",       True),
    ({"book", "clock", "vase"},                                                  "Interior",              True),
]

def _infer_scene(labels: list[str]) -> dict:
    label_set = set(l.lower() for l in labels)
    for objects, scene, indoor in _SCENE_RULES:
        if objects & label_set:
            return {
                "label":    scene,
                "indoor":   indoor,
                "matched":  list(objects & label_set),
            }
    return {"label": "Unknown", "indoor": None, "matched": []}


# ── FACE DETECTION ────────────────────────────────────────
def detect_faces(frame_rgb) -> list[dict]:
    """Returns list of {x,y,w,h,confidence} dicts. frame_rgb is RGB numpy array."""
    if _FACE_DET is None:
        return []
    try:
        import cv2
        import numpy as np

        # Try MediaPipe first
        try:
            import mediapipe as mp
            results = _FACE_DET.process(frame_rgb)
            if not results.detections:
                return []
            h, w = frame_rgb.shape[:2]
            faces = []
            for det in results.detections:
                bb  = det.location_data.relative_bounding_box
                fx  = max(0, int(bb.xmin * w))
                fy  = max(0, int(bb.ymin * h))
                fw  = int(bb.width * w)
                fh  = int(bb.height * h)
                faces.append({
                    "x": fx, "y": fy, "w": fw, "h": fh,
                    "confidence": round(det.score[0], 3),
                    "id": len(faces),
                })
            return faces
        except Exception:
            pass

        # OpenCV Haar fallback
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        dets = _FACE_DET.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [{"x":int(x),"y":int(y),"w":int(w),"h":int(h),"confidence":0.8,"id":i}
                for i,(x,y,w,h) in enumerate(dets)]
    except Exception as e:
        log.debug(f"[Analyzer] Face detection error: {e}")
        return []


# ── OBJECT DETECTION ──────────────────────────────────────
def detect_objects(frame_rgb) -> tuple[list[dict], dict]:
    """Returns (objects_list, scene_dict). Objects: {label,x,y,w,h,confidence}."""
    if _YOLO is None:
        return [], {"label": "Unknown", "indoor": None}
    try:
        import numpy as np
        results = _YOLO(frame_rgb, verbose=False)[0]
        objects = []
        labels  = []
        for box in results.boxes:
            cls   = int(box.cls[0])
            label = _YOLO.names[cls]
            conf  = float(box.conf[0])
            if conf < 0.35:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append({
                "label":      label,
                "x":          int(x1), "y": int(y1),
                "w":          int(x2 - x1), "h": int(y2 - y1),
                "confidence": round(conf, 3),
            })
            labels.append(label)
        scene = _infer_scene(labels)
        return objects, scene
    except Exception as e:
        log.debug(f"[Analyzer] YOLO error: {e}")
        return [], {"label": "Unknown", "indoor": None}


# ── OCR TEXT DETECTION ────────────────────────────────────
def detect_text(frame_rgb) -> list[str]:
    """Returns list of detected text strings (confidence > 0.4)."""
    if _OCR is None:
        return []
    try:
        import cv2
        # Resize for speed
        h, w = frame_rgb.shape[:2]
        scale = min(1.0, 640 / max(w, h))
        if scale < 1.0:
            frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
        results = _OCR.readtext(frame_rgb, detail=1)
        return [text for (_, text, conf) in results if conf > 0.4 and len(text.strip()) > 1]
    except Exception as e:
        log.debug(f"[Analyzer] OCR error: {e}")
        return []


# ── SPEECH TRANSCRIPTION ─────────────────────────────────
def transcribe_audio(video_path: str) -> Optional[dict]:
    """
    Transcribes audio from the video using Whisper.
    Returns {"text": "...", "segments": [...]} or None.
    This is run ONCE in a background thread — not per frame.
    """
    if _WHISPER is None or not os.path.exists(video_path):
        return None
    try:
        log.info(f"[Analyzer] Transcribing audio: {video_path}")
        result = _WHISPER.transcribe(
            video_path,
            language="en",
            fp16=False,
            verbose=False,
            word_timestamps=True,
        )
        log.info(f"[Analyzer] Transcription complete — {len(result.get('segments', []))} segments")
        return {
            "text":     result.get("text", "").strip(),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
        }
    except Exception as e:
        log.warning(f"[Analyzer] Whisper failed: {e}")
        return None


# ── ACTION INFERENCE ──────────────────────────────────────
_ACTION_RULES = [
    ({"person", "sports ball", "tennis racket", "baseball bat"}, "Playing sport"),
    ({"person", "laptop", "keyboard"},                           "Working / typing"),
    ({"person", "microphone", "tie"},                            "Speaking / presenting"),
    ({"person", "phone"},                                         "Using phone"),
    ({"person", "car", "truck"},                                  "Driving / riding"),
    ({"person", "dining table", "fork"},                          "Eating"),
    ({"person", "book"},                                          "Reading"),
    ({"person", "tv"},                                            "Watching TV"),
]

def infer_action(objects: list[dict]) -> Optional[str]:
    labels = set(o["label"].lower() for o in objects)
    for required, action in _ACTION_RULES:
        if required.issubset(labels):
            return action
    if "person" in labels:
        n = sum(1 for o in objects if o["label"] == "person")
        return f"{n} person{'s' if n>1 else ''} in scene"
    return None
