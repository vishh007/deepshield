# DeepShield v3

Real-time deepfake and AI-generated video detection. Upload a file or paste a social media URL — the system streams frame-by-frame analysis live as the video plays, with face detection, scene intelligence, OCR, and speech transcription.

---

## ⚠️ Python 3.11 Required

`torch`, `mediapipe`, and `easyocr` have broken wheels on Python 3.12+. This will not work on newer Python versions.

### Step 1 — Check your current version

```bash
python --version
```

If it says **3.12 or higher**, follow the steps below to install 3.11 alongside it. You do **not** need to uninstall your current Python.

---

### Step 2 — Install Python 3.11

Download the installer for your OS:

**Windows:** https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
- Run the installer
- ✅ Check **"Add Python to PATH"**
- Click **Install Now**

**Mac:** https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg
- Run the `.pkg` installer

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

After installing, verify:
```bash
py -3.11 --version       # Windows
python3.11 --version     # Mac/Linux
```

It should print `Python 3.11.9`.

---

## Setup

### Step 3 — Create a virtual environment with Python 3.11

```bash
# Windows
py -3.11 -m venv venv

# Mac/Linux
python3.11 -m venv venv
```

### Step 4 — Activate the environment

You must activate the environment **every time** you open a new terminal.

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Mac/Linux
source venv/bin/activate
```

> **Windows PowerShell error?** Run this once to allow scripts:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

When activated, your terminal prompt will show `(venv)` at the start.

### Step 5 — Install dependencies

```bash
pip install -r requirements.txt
```

This takes 5–15 minutes depending on your connection. torch alone is ~2GB.

### Step 6 — Download the ML models

Run this once. Models are not included in the repo.

```bash
cd backend
python download_models.py
```

This downloads:
| Model | Size | Purpose |
|---|---|---|
| umm-maybe/AI-image-detector | ~350MB | Core deepfake detection (Vision Transformer) |
| YOLOv8n | ~6MB | Object + scene detection |
| Whisper tiny | ~150MB | Speech transcription |

Models are cached in `~/.cache/huggingface/` and `~/.cache/whisper/` — they only download once.

### Step 7 — Start the server

```bash
# Make sure you're in the backend/ folder
cd backend
uvicorn main:app --reload --port 8000
```

First startup takes **60–90 seconds** while the 350MB model loads into RAM — this is normal.

Open **http://localhost:8000** in your browser.

---

## Optional: Groq AI Reports

Enables AI-written forensic reports after each scan. Free at console.groq.com (no credit card, 14,400 requests/day).

```bash
# Windows (PowerShell) — set before starting the server
$env:GROQ_API_KEY = "gsk_your_key_here"

# Mac/Linux
export GROQ_API_KEY="gsk_your_key_here"
```

Falls back to a template report if no key is set — everything else still works.

---

## What to Upload to Git

### ✅ Upload these files

```
deepshield-v3/
├── backend/
│   ├── main.py
│   ├── ml_engine.py
│   ├── decision_engine.py
│   ├── ws_streamer.py
│   ├── video_analyzer.py
│   ├── video_downloader.py
│   ├── report_generator.py
│   └── download_models.py
├── static/
│   └── index.html
├── requirements.txt
├── README.md
└── .gitignore
```

### ❌ Do NOT upload these

| What | Why |
|---|---|
| `venv/` | Virtual environment — 2GB+, rebuilds from requirements.txt |
| `*.pt`, `*.pth`, `*.bin`, `*.safetensors` | ML model weights — download_models.py fetches them |
| `backend/tmp_videos/` | Temporary video files from scans |
| `*.mp4`, `*.avi`, `*.mov` | Video files |
| `.env` | Contains secrets (API keys) |
| `__pycache__/` | Python bytecode cache |
| `.vscode/`, `.idea/` | IDE settings |
| `~/.cache/huggingface/` | Model cache — outside your project folder, not in git |

Everything in the "do not upload" list is already handled by `.gitignore`.

---

## Pushing to GitHub

```bash
# 1. Create a NEW empty repo on github.com (no README, no gitignore)

# 2. From your project root folder (where README.md is):
git init
git add .
git commit -m "initial commit"

# 3. Link to your GitHub repo (replace with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/deepshield-v3.git
git branch -M main
git push -u origin main
```

After that, to push new changes:
```bash
git add .
git commit -m "describe what you changed"
git push
```

---

## After Cloning on a New Machine

```bash
git clone https://github.com/YOUR_USERNAME/deepshield-v3.git
cd deepshield-v3

# Install Python 3.11 (see Step 2 above)
py -3.11 -m venv venv          # Windows
python3.11 -m venv venv        # Mac/Linux

.\venv\Scripts\Activate.ps1    # Windows
source venv/bin/activate        # Mac/Linux

pip install -r requirements.txt
cd backend
python download_models.py
uvicorn main:app --reload --port 8000
```

---

## Common Issues

**`Activate.ps1` blocked** — run once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**`mediapipe` / `easyocr` install fails** — you're not on Python 3.11. Check `python --version` inside the activated venv.

**Slow startup (60–90s)** — normal, the 350MB model loads into RAM once per server start.

**Slow analysis (30–60s per video)** — normal on CPU. GPU cuts it to ~5s. If you have CUDA, set `device=0` in `ml_engine.py` line ~20.

**Edited videos getting BLOCKED** — the auto-calibration in `decision_engine.py` handles this. If you still get false positives, raise `AVG_HIGH` in the `edited` threshold block by 0.03.

**Port 8000 already in use** — change the port: `uvicorn main:app --reload --port 8001` then open `localhost:8001`.
