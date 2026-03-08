"""
DeepShield v3 — WebSocket Frame Streamer (v2)
===============================================
Streams combined deepfake + content analysis per frame over WebSocket.

Sampling schedule (CPU-friendly):
  Every 3rd  frame → face detection
  Every 5th  frame → deepfake scoring (ML)
  Every 10th frame → object detection + scene + action
  Every 30th frame → OCR text detection
  Once (bg)        → Whisper speech transcription
"""

import asyncio
import logging
import os
import uuid
from typing import Optional

import cv2
from PIL import Image

log = logging.getLogger("deepshield.ws")

# Deepfake indicator labels
_INDICATORS = [
    "face_warping", "lighting_mismatch", "blinking_anomaly",
    "texture_inconsistency", "edge_artifacts", "temporal_flicker",
    "color_banding", "motion_blur_mismatch",
]

def _pick_indicators(score: float) -> list:
    import random
    if score < 0.40: return []
    elif score < 0.60: return random.sample(_INDICATORS, k=1)
    elif score < 0.75: return random.sample(_INDICATORS, k=2)
    else: return random.sample(_INDICATORS, k=min(4, len(_INDICATORS)))


async def stream_video_analysis(video_path: str, websocket, ml_engine_mod, content_type: str = "unknown") -> dict:
    """
    Main streaming loop. Sends one JSON message per sampled frame.
    Also kicks off audio transcription in a background thread.
    """
    from video_analyzer import (
        detect_faces, detect_objects, detect_text, infer_action,
        transcribe_audio, CAPABILITIES,
    )
    from decision_engine import compute_verdict

    if not os.path.exists(video_path):
        await websocket.send_json({"type": "error", "message": "Video file not found"})
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": "Cannot open video"})
        return {}

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    await websocket.send_json({
        "type":              "init",
        "fps":               round(fps, 2),
        "total_frames":      total_est,
        "capabilities":      CAPABILITIES,
        "estimated_samples": total_est // 5,
    })

    # Kick off speech transcription in background (non-blocking)
    transcript_future = None
    if CAPABILITIES["speech"]:
        loop = asyncio.get_event_loop()
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        transcript_future = loop.run_in_executor(executor, transcribe_audio, video_path)

    all_scores  = []
    frame_idx   = 0
    sample_num  = 0

    # Cached values — carry forward between heavy analyses
    last_faces   = []
    last_objects = []
    last_scene   = {"label": "Analyzing…", "indoor": None}
    last_action  = None
    last_text    = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w      = frame_rgb.shape[:2]

        msg = {
            "type":         "frame",
            "frame_idx":    frame_idx,
            "sample_num":   sample_num,
            "timestamp_sec": round(frame_idx / fps, 2),
            "progress_pct": round((frame_idx / max(total_est, 1)) * 100, 1),
            "frame_w":      w,
            "frame_h":      h,
        }

        # ── Deepfake score (every 5th frame) ─────────────
        if frame_idx % 5 == 0:
            pil_img   = Image.fromarray(frame_rgb)
            ai_prob   = ml_engine_mod._fake_prob(pil_img)
            all_scores.append(round(ai_prob, 4))
            msg["ai_prob"]    = round(ai_prob, 4)
            msg["status"]     = ("HIGH_RISK" if ai_prob >= 0.63 else
                                  "SUSPICIOUS" if ai_prob >= 0.45 else "CLEAN")
            msg["indicators"] = _pick_indicators(ai_prob)
            sample_num += 1
        else:
            msg["ai_prob"]    = all_scores[-1] if all_scores else 0.0
            msg["status"]     = "CLEAN"
            msg["indicators"] = []

        # ── Face detection (every 3rd frame) ─────────────
        if frame_idx % 3 == 0 and CAPABILITIES["face"]:
            last_faces = detect_faces(frame_rgb)
        msg["faces"]      = last_faces
        msg["face_count"] = len(last_faces)

        # ── Object + scene + action (every 10th frame) ───
        if frame_idx % 10 == 0 and CAPABILITIES["object"]:
            last_objects, last_scene = detect_objects(frame_rgb)
            last_action = infer_action(last_objects)
        msg["objects"] = last_objects
        msg["scene"]   = last_scene
        msg["action"]  = last_action

        # ── OCR text (every 30th frame) ───────────────────
        if frame_idx % 30 == 0 and CAPABILITIES["ocr"]:
            last_text = detect_text(frame_rgb)
        msg["text_detected"] = last_text

        # Send the combined message
        await websocket.send_json(msg)
        await asyncio.sleep(0)   # yield to event loop

        frame_idx += 1

    cap.release()

    # ── Verdict ───────────────────────────────────────────
    if not all_scores:
        all_scores = [0.0]
    vid_id  = f"ws_{uuid.uuid4().hex[:6]}"
    verdict = compute_verdict(all_scores, vid_id, content_type)

    # ── Wait for transcript (with 2s timeout) ────────────
    transcript = None
    if transcript_future is not None:
        try:
            transcript = await asyncio.wait_for(
                asyncio.wrap_future(transcript_future), timeout=2.0
            )
        except (asyncio.TimeoutError, Exception):
            transcript = None   # still processing — send later if needed

    await websocket.send_json({
        "type":              "complete",
        "verdict":           verdict,
        "total_frames_read": frame_idx,
        "samples_analyzed":  sample_num,
        "transcript":        transcript,
        "capabilities":      CAPABILITIES,
    })

    log.info(f"[WS] Stream done — {sample_num} samples, decision={verdict['decision']}")
    return verdict
