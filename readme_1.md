# Whisper Batch Transcription (running on Jetstream2 / Linux Debian)

Repo documents our **batch transcription** of YT Short using **Whisper large-v3** on a Jetstream2 GPU A100 instance.

## Features
- Recursively finds common video formats (`.mp4, .webm, .mkv, .mov, .m4v, .avi`)
- Parallel processing (configurable), idempotent (safe to re-run)
- Outputs `.txt`, `.srt`, `.vtt`, `.tsv`, `.json`
- Handles spaces and special characters in filenames
- Optional language hint (`--language XX`) to avoid auto-detect issues

---

## 1. System Requirements
- Ubuntu/Debian (tested on Ubuntu 22.04)
- NVIDIA GPU (A100 on Jetstream2) + CUDA driver
- `ffmpeg` (system package)

---

## 2. Create a Python venv (no conda)
```bash
python3 -m venv ~/whisper-venv
source ~/whisper-venv/bin/activate
pip install --upgrade pip
