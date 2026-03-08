"""
DeepShield v3 — Decision Engine
=================================
Auto-calibrates thresholds based on score distribution patterns.

Key insight:
  Edited real video  → HIGH avg, LOW variance, LOW spikes (uniformly elevated)
  Real deepfake      → MEDIUM avg, HIGH variance, HIGH localized spikes
  AI-generated video → HIGH avg, MEDIUM variance, CONSISTENT high scores
  Real natural video → LOW avg, LOW variance

We auto-detect "edited" pattern and adjust thresholds before scoring.
"""

from typing import List, Dict, Any, Tuple
import statistics


# ── Base thresholds ───────────────────────────────────────
# These are adjusted dynamically based on auto-detected content type
_BASE = {
    "natural": {
        "AVG_HIGH": 0.63, "AVG_MED": 0.50,
        "SUSP_HIGH": 0.40, "SUSP_MED": 0.18,
        "TEMP_HIGH": 0.63, "TEMP_MED": 0.38,
        "SPIKE_HIGH": 4,   "SPIKE_MED": 2,
        "SPIKE_VAL": 0.78,
    },
    "edited": {
        # Edited/graded video: colour LUTs, VFX, hard cuts all inflate scores
        # Require much stronger and more localised signal to BLOCK
        "AVG_HIGH": 0.88, "AVG_MED": 0.76,
        "SUSP_HIGH": 0.72, "SUSP_MED": 0.50,
        "TEMP_HIGH": 0.85, "TEMP_MED": 0.68,
        "SPIKE_HIGH": 10,  "SPIKE_MED": 5,
        "SPIKE_VAL": 0.95,
    },
    "natural_ai": {
        # Genuine AI-generated video (Sora, Pika, RunwayML)
        # Scores are uniformly high AND have specific face/texture anomalies
        "AVG_HIGH": 0.68, "AVG_MED": 0.55,
        "SUSP_HIGH": 0.45, "SUSP_MED": 0.22,
        "TEMP_HIGH": 0.68, "TEMP_MED": 0.42,
        "SPIKE_HIGH": 5,   "SPIKE_MED": 2,
        "SPIKE_VAL": 0.82,
    },
}


def _auto_detect_type(scores: List[float]) -> Tuple[str, dict]:
    """
    Infer whether the video is edited real footage or AI-generated
    by analysing the statistical distribution of frame scores.

    Edited real footage profile:
      - High avg (>0.65) BUT low stddev (<0.08)
      - Scores are uniformly elevated — editing affects every frame equally
      - Very few extreme spikes above 0.90

    AI-generated video profile:
      - High avg (>0.65) AND medium-high stddev (>0.08)
      - Localized spikes — model detects specific faces/textures
      - OR consistently very high (>0.85) across all frames

    Natural video:
      - Low avg (<0.55) and low stddev
    """
    n = len(scores)
    if n < 3:
        return "natural_ai", _BASE["natural_ai"]

    avg = sum(scores) / n
    stddev = statistics.stdev(scores) if n > 1 else 0.0
    spikes_90 = sum(1 for s in scores if s > 0.90)
    spikes_80 = sum(1 for s in scores if s > 0.80)
    pct_above_70 = sum(1 for s in scores if s > 0.70) / n

    # --- Pattern 1: Edited real footage ---
    # High avg + very uniform scores (low stddev) + very few extreme spikes
    # The model sees "unnatural" colour but can't find specific deepfake regions
    if avg > 0.65 and stddev < 0.09 and spikes_90 < (n * 0.05):
        return "edited", _BASE["edited"]

    # --- Pattern 2: Borderline edited (moderate avg, very uniform) ---
    if avg > 0.58 and stddev < 0.07:
        return "edited", _BASE["edited"]

    # --- Pattern 3: Natural real video ---
    if avg < 0.52 and stddev < 0.12:
        return "natural", _BASE["natural"]

    # --- Default: treat as possible AI-generated ---
    return "natural_ai", _BASE["natural_ai"]


def _sig(v, med, hi):
    return "HIGH" if v >= hi else "MEDIUM" if v >= med else "LOW"


def _temporal(scores):
    if len(scores) < 2:
        return scores[0] if scores else 0.0
    avg = sum(scores) / len(scores)
    return round(min(1.0, avg * 0.7 + statistics.stdev(scores) * 0.3), 4)


def compute_verdict(
    frame_scores: List[float],
    video_id: str,
    content_type: str = "auto",
) -> Dict[str, Any]:

    n   = len(frame_scores)
    avg = round(sum(frame_scores) / n, 4)
    max_s = round(max(frame_scores), 4)

    # Auto-detect content type from score distribution
    if content_type in ("auto", "unknown"):
        detected_type, t = _auto_detect_type(frame_scores)
    else:
        detected_type = content_type
        t = _BASE.get(content_type, _BASE["natural_ai"])

    spikes   = sum(1 for s in frame_scores if s > t["SPIKE_VAL"])
    susp     = sum(1 for s in frame_scores if s > 0.60)
    susp_pct = round(susp / n, 4)
    temporal = _temporal(frame_scores)

    signals = {
        "avg_score":  _sig(avg,      t["AVG_MED"],  t["AVG_HIGH"]),
        "suspicious": _sig(susp_pct, t["SUSP_MED"], t["SUSP_HIGH"]),
        "temporal":   _sig(temporal, t["TEMP_MED"], t["TEMP_HIGH"]),
        "spikes":     _sig(spikes,   t["SPIKE_MED"], t["SPIKE_HIGH"]),
    }

    highs = sum(1 for v in signals.values() if v == "HIGH")
    meds  = sum(1 for v in signals.values() if v == "MEDIUM")

    type_notes = {
        "edited":     "Score pattern matches edited/colour-graded footage — relaxed thresholds applied.",
        "natural":    "Score pattern matches natural unedited footage.",
        "natural_ai": "Score pattern consistent with AI-generated or deepfake content.",
    }
    calibration_note = type_notes.get(detected_type, "")

    if highs >= 2:
        decision, risk = "BLOCKED", "HIGH"
        reason = (
            f"Strong AI-generation signals across {susp}/{n} frames. "
            f"Avg: {avg:.0%}. {calibration_note}"
        )
        action = "Remove content immediately. Flag account for audit."
    elif highs == 1 or meds >= 2:
        decision, risk = "FLAGGED", "MEDIUM"
        reason = (
            f"Suspicious AI patterns in {susp}/{n} frames. "
            f"Avg: {avg:.0%}. Review recommended. {calibration_note}"
        )
        action = "Hold from feed. Queue for human review."
    else:
        decision, risk = "APPROVED", "LOW"
        reason = (
            f"No significant AI-generation signals. "
            f"Avg: {avg:.0%} across {n} frames. {calibration_note}"
        )
        action = "Allow to publish."

    return {
        "video_id":           video_id,
        "decision":           decision,
        "risk_level":         risk,
        "reason":             reason,
        "recommended_action": action,
        "content_type":       detected_type,
        "metadata": {
            "avg_fake_probability": avg,
            "max_fake_probability": max_s,
            "frames_analyzed":      n,
            "suspicious_frames":    susp,
            "suspicious_pct":       round(susp_pct * 100, 1),
            "spike_count":          spikes,
            "temporal_score":       temporal,
            "stddev":               round(statistics.stdev(frame_scores) if n > 1 else 0, 4),
            "auto_detected_type":   detected_type,
            "signals":              signals,
        },
    }
