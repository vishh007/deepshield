"""
DeepShield — AI Report Generator
Uses Groq (free) to generate a short forensic report from ML scores.
Get free API key at: https://console.groq.com
"""

import logging
import os
from typing import Dict, Any

log = logging.getLogger("deepshield.report")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def generate_report(verdict: Dict[str, Any], source_info: str = "") -> str:
    """
    Send ML verdict to Groq and get a human-readable forensic report.
    Falls back to a smart template if Groq key is not set.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_key_here":
        return _template_report(verdict, source_info)

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        m = verdict.get("metadata", {})
        prompt = f"""You are a forensic AI analyst. Analyze this deepfake/AI-content detection result and write a SHORT 3-4 sentence professional report.

Detection Results:
- Decision: {verdict.get('decision')}
- Risk Level: {verdict.get('risk_level')}
- Average AI probability: {m.get('avg_fake_probability', 0):.1%}
- Max AI probability: {m.get('max_fake_probability', 0):.1%}
- Frames analyzed: {m.get('frames_analyzed', 0)}
- Suspicious frames: {m.get('suspicious_frames', 0)} ({m.get('suspicious_pct', 0)}%)
- High-confidence spikes: {m.get('spike_count', 0)}
- Temporal consistency score: {m.get('temporal_score', 0):.3f}
- ML Signals: {m.get('signals', {})}
- Source: {source_info or 'uploaded file'}

Write a concise forensic report. Be specific about the numbers. Do not use bullet points. Plain paragraph only. Do not start with 'I'."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        report = response.choices[0].message.content.strip()
        log.info("[Report] Groq report generated successfully")
        return report

    except Exception as e:
        log.warning(f"[Report] Groq failed ({e}), using template")
        return _template_report(verdict, source_info)


def _template_report(verdict: Dict[str, Any], source_info: str) -> str:
    """Smart template fallback when Groq is unavailable."""
    m   = verdict.get("metadata", {})
    dec = verdict.get("decision", "UNKNOWN")
    avg = m.get("avg_fake_probability", 0)
    sus = m.get("suspicious_pct", 0)
    frm = m.get("frames_analyzed", 0)
    spk = m.get("spike_count", 0)
    src = source_info or "the uploaded file"

    if dec == "BLOCKED":
        return (
            f"Analysis of {src} returned a BLOCKED verdict with high confidence. "
            f"The model flagged {avg:.1%} average AI-generation probability across {frm} analyzed frames, "
            f"with {sus}% of frames showing suspicious patterns and {spk} high-confidence spike(s) above 80%. "
            f"These indicators are strongly consistent with AI-generated or synthetically manipulated content. "
            f"Immediate review and content removal is recommended."
        )
    elif dec == "FLAGGED":
        return (
            f"Analysis of {src} returned a FLAGGED verdict requiring human review. "
            f"The model detected an average AI probability of {avg:.1%} across {frm} frames, "
            f"with {sus}% of frames exhibiting anomalous patterns. "
            f"While not conclusive, these signals warrant further investigation before the content is approved. "
            f"Manual verification by a trained reviewer is advised."
        )
    else:
        return (
            f"Analysis of {src} returned an APPROVED verdict with low AI-generation probability. "
            f"The model scored an average of {avg:.1%} across {frm} analyzed frames, "
            f"with minimal suspicious activity detected ({sus}% of frames). "
            f"No significant deepfake or AI-generation signals were identified. "
            f"Content appears authentic based on current model parameters."
        )
