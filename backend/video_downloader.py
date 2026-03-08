"""
DeepShield — Video Downloader
Downloads videos from social media URLs using yt-dlp.
"""
import os, subprocess, uuid
from typing import Optional, Tuple

SUPPORTED = [
    "instagram.com","instagr.am","tiktok.com","vm.tiktok.com",
    "youtube.com","youtu.be","facebook.com","fb.watch","fb.com",
    "twitter.com","x.com","t.co","reddit.com","redd.it",
    "snapchat.com","twitch.tv","vimeo.com","linkedin.com",
    "dailymotion.com","rumble.com",
]
MAX_DURATION = 180
MAX_SIZE     = "200M"


def detect_platform(url: str) -> str:
    u = url.lower()
    for domain in SUPPORTED:
        if domain in u:
            return domain.split(".")[0].capitalize()
    return "Unknown"


def download_video(url: str, output_dir: str = "tmp_videos", output_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Download video from URL.
    If output_path is given, save directly to that path.
    Otherwise save to output_dir with auto-generated name.
    """
    os.makedirs(output_dir, exist_ok=True)
    platform = detect_platform(url)

    if output_path:
        # Save directly to specified path
        out_tmpl = output_path.replace(".mp4", ".%(ext)s")
    else:
        vid_id   = uuid.uuid4().hex[:10]
        out_tmpl = os.path.join(output_dir, f"{vid_id}.%(ext)s")

    cmd = [
        "yt-dlp", "--no-playlist",
        "--max-filesize", MAX_SIZE,
        "--match-filter", f"duration < {MAX_DURATION}",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-warnings", "-o", out_tmpl, url,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        raise ValueError("Download timed out after 120 seconds.")

    if proc.returncode != 0:
        err = next((l for l in proc.stderr.splitlines() if "ERROR" in l), proc.stderr[:300])
        raise ValueError(f"yt-dlp failed: {err}")

    # Find the output file
    if output_path and os.path.exists(output_path):
        return output_path, platform

    search_dir = os.path.dirname(out_tmpl)
    prefix     = os.path.basename(out_tmpl).split(".")[0]
    for fname in os.listdir(search_dir):
        if fname.startswith(prefix):
            return os.path.join(search_dir, fname), platform

    raise ValueError("Download finished but output file was not found.")


def cleanup(path: Optional[str]) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
