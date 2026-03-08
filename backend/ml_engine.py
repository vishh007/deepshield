"""
DeepShield v3 — ML Engine
==========================
Model: umm-maybe/AI-image-detector
Trained on: DALL-E, Midjourney, Stable Diffusion, and other AI generators.
Detects AI-GENERATED content broadly — not just face swaps.

Labels: {0: 'artificial', 1: 'human'}
  artificial = AI-generated
  human      = real/authentic
"""

import logging
import os
from typing import List, Tuple

import cv2
import torch
from PIL import Image
from transformers import pipeline

log = logging.getLogger("deepshield.ml")

_CLASSIFIER  = None
_FAKE_LABEL  = None   # the label that means "AI-generated"


def load_model() -> None:
    """Call ONCE at startup."""
    global _CLASSIFIER, _FAKE_LABEL

    model_id    = "umm-maybe/AI-image-detector"
    device      = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"

    log.info(f"[ML] Loading '{model_id}' on {device_name}...")

    _CLASSIFIER = pipeline(
        task="image-classification",
        model=model_id,
        device=device,
    )

    id2label = _CLASSIFIER.model.config.id2label
    log.info(f"[ML] Label map: {id2label}")

    # Find the label that means AI/fake/artificial/generated
    _FAKE_LABEL = next(
        (lbl for lbl in id2label.values()
         if any(k in lbl.lower() for k in ["fake","artificial","ai","generated","synthetic","deepfake"])),
        None
    )

    if _FAKE_LABEL is None:
        # If still not found, assume lower index is real, higher is fake
        _FAKE_LABEL = id2label[max(id2label.keys())]
        log.warning(f"[ML] Could not auto-detect fake label — defaulting to '{_FAKE_LABEL}'")

    log.info(f"[ML] AI-generated label: '{_FAKE_LABEL}'")
    log.info("[ML] Model ready.")


def _fake_prob(img: Image.Image) -> float:
    """Score one frame. Returns AI-generated probability in [0, 1]."""
    if _CLASSIFIER is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    results: List[dict] = _CLASSIFIER(img)

    for r in results:
        if r["label"] == _FAKE_LABEL:
            return float(r["score"])

    # Fallback — infer from top result
    top      = results[0]
    is_fake  = any(k in top["label"].lower() for k in ["fake","artificial","ai","generated","synthetic"])
    return float(top["score"]) if is_fake else float(1 - top["score"])


def analyze_video(video_path: str, sample_every: int = 5) -> Tuple[List[float], int]:
    """
    Extract frames, score each one for AI-generation probability.

    Args:
        video_path:   Path to video file.
        sample_every: Analyze every Nth frame (default every 5th for speed).

    Returns:
        (frame_scores, total_frames_read)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    scores: List[float] = []
    idx = 0

    log.info(f"[ML] Analyzing '{os.path.basename(video_path)}' for AI-generated content...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prob = _fake_prob(img)
            scores.append(round(prob, 4))
            log.debug(f"[ML]   frame {idx:4d}  ai_prob={prob:.4f}")
        idx += 1

    cap.release()

    if not scores:
        log.warning("[ML] No frames sampled.")
        scores = [0.0]

    avg = sum(scores) / len(scores)
    log.info(f"[ML] Done — {len(scores)} frames, avg_ai={avg:.4f}, max_ai={max(scores):.4f}")
    return scores, idx
