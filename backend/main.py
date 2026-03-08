"""
DeepShield v3 — FastAPI + WebSocket
URL mode now: download first → WebSocket stream (real-time)
"""
import logging, os, shutil, uuid, asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import ml_engine
from decision_engine import compute_verdict
from video_downloader import cleanup, detect_platform, download_video
from report_generator import generate_report
from ws_streamer import stream_video_analysis
from video_analyzer import load_analyzers, CAPABILITIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("deepshield.api")
results:   dict = {}
audit:     list = []
downloads: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 55)
    log.info("DeepShield v3 starting up...")
    ml_engine.load_model()
    load_analyzers()
    log.info(f"Capabilities: {CAPABILITIES}")
    log.info("Startup complete. Ready.")
    log.info("=" * 55)
    yield
    log.info("Shutting down.")


app = FastAPI(title="DeepShield API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


class UrlRequest(BaseModel):
    url: str
    content_type: str = "unknown"   # "natural" | "edited" | "unknown"
    model_config = {"extra": "forbid"}
    @field_validator("url")
    @classmethod
    def must_be_http(cls, v):
        if not v.strip().startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v.strip()


# ── WebSocket: streams any already-downloaded file ────────
@app.websocket("/ws/stream/{video_id}")
async def websocket_stream(websocket: WebSocket, video_id: str):
    await websocket.accept()
    log.info(f"[WS] Client connected: {video_id}")
    video_path = f"tmp_videos/{video_id}.mp4"

    # Wait up to 120s for file (URL mode downloads first)
    for i in range(120):
        info = results.get(video_id, {})
        if info.get("status") == "ERROR":
            await websocket.send_json({"type": "error", "message": info.get("error", "Download failed")})
            await websocket.close()
            return
        if os.path.exists(video_path):
            break
        await asyncio.sleep(1)

    if not os.path.exists(video_path):
        await websocket.send_json({"type": "error", "message": "Video not found — download may have failed"})
        await websocket.close()
        return

    content_type = results.get(video_id, {}).get("content_type", "unknown")

    try:
        verdict = await stream_video_analysis(video_path, websocket, ml_engine, content_type)
        if verdict:
            source = results.get(video_id, {}).get("source", "file upload")
            report_text = generate_report(verdict, source)
            verdict.update({
                "status": "COMPLETE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "can_download": video_id in downloads,
                "ai_report": report_text,
            })
            results[video_id] = {**results.get(video_id, {}), **verdict}
            audit.append(verdict)
    except WebSocketDisconnect:
        log.info(f"[WS] Disconnected: {video_id}")
    except Exception as e:
        log.error(f"[WS] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Keep file if it was a URL download (so user can play it)
        if video_id not in downloads:
            cleanup(video_path)
        log.info(f"[WS] Stream closed: {video_id}")


# ── Upload file for WS stream ──────────────────────────────
@app.post("/api/upload/stream")
async def upload_for_stream(file: UploadFile = File(...)):
    video_id = f"vid_{uuid.uuid4().hex[:8]}"
    os.makedirs("tmp_videos", exist_ok=True)
    tmp = f"tmp_videos/{video_id}.mp4"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    results[video_id] = {"video_id": video_id, "status": "READY", "source": file.filename or "upload", "content_type": "unknown"}
    log.info(f"[API] File uploaded: {video_id}")
    return {"video_id": video_id, "filename": file.filename}


# ── URL: download then stream via WS ──────────────────────
@app.post("/api/analyze/url")
async def analyze_url(req: UrlRequest, bg: BackgroundTasks):
    video_id = f"vid_{uuid.uuid4().hex[:8]}"
    platform = detect_platform(req.url)
    results[video_id] = {
        "video_id": video_id,
        "status": "DOWNLOADING",
        "platform": platform,
        "source": req.url,
        "content_type": req.content_type,
        "original_url": req.url,
    }
    log.info(f"[{video_id}] URL queued — platform={platform} type={req.content_type}")

    async def _download():
        try:
            os.makedirs("tmp_videos", exist_ok=True)
            tmp = f"tmp_videos/{video_id}.mp4"
            actual, _ = download_video(req.url, output_path=tmp)
            downloads[video_id] = actual
            results[video_id]["status"] = "READY"
            log.info(f"[{video_id}] Download complete → {actual}")
        except Exception as e:
            log.error(f"[{video_id}] Download failed: {e}")
            results[video_id]["status"] = "ERROR"
            results[video_id]["error"] = str(e)

    bg.add_task(_download)
    return {"video_id": video_id, "status": "DOWNLOADING", "platform": platform}


@app.get("/api/result/{video_id}")
def get_result(video_id: str):
    e = results.get(video_id)
    if not e:
        raise HTTPException(404, "Not found")
    return e


@app.get("/api/download/{video_id}")
def dl(video_id: str):
    p = downloads.get(video_id)
    if not p or not os.path.exists(p):
        raise HTTPException(404, "Not available")
    return FileResponse(p, media_type="video/mp4", filename=f"deepshield_{video_id}.mp4")


@app.get("/api/report/{video_id}")
def report(video_id: str):
    e = results.get(video_id)
    if not e:
        raise HTTPException(404, "Not found")
    return {"video_id": video_id, "report": e.get("ai_report", ""), "decision": e.get("decision"), "timestamp": e.get("timestamp")}


@app.get("/health")
def health():
    return {
        "status": "ok", "version": "3.0.0",
        "model_loaded": ml_engine._CLASSIFIER is not None,
        "fake_label": ml_engine._FAKE_LABEL,
        "capabilities": CAPABILITIES,
        "cached": len(results),
    }


@app.get("/")
def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "DeepShield API v3"}


@app.get("/api/audit")
def get_audit(limit: int = 50):
    return {"total": len(audit), "entries": audit[-limit:]}
