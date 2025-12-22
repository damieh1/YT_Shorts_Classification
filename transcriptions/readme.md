# Whisper Batch Transcription (running on Jetstream2 / Linux Debian)

All transcription scripts are stored in [transcriptions/](/transcriptions). 
- [transcriptions/transcribe_dir.sh](/transcriptions/transcribe_dir.sh): batch transcription wrapper
- [transcriptions/check_env.py](/transcriptions/env_check.py): verify environment and versions
- [See synthetic transcript structure](/synthetic_data/example_transcript.json): json output

--- 


Repo documents our **batch transcription** of YT Short using **Whisper large-v3** on a Jetstream2 GPU A100 instance.
To ensure smooth, error-free execution, check the [`requirements`](requirements.txt). 


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
```

---

## 3. Get correct CUDA 
- Install CUDA 12.1
```
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

sudo apt update && sudo apt install -y ffmpeg
```

## 4. Run Whisper on Corpus
- Excutable script
```
chmod +x transcribe_dir.sh
```
- Run Whisper
```
./transcribe_dir.sh "/path/to/videos" "/path/to/output_captions" 4
```
- Arg1: input directory
- Arg2: output directory-
- Arg3: number of parallel jobs (use 2–4 for Whisper large-v3 on A100)

Create five outputs: `.txt, .srt, .vtt, .tsv, .json`

---

## 5. Monitor GPU
```
watch -n 1 nvidia-smi
ls /path/to/output_captions | grep '\.srt$' | wc -l
```

## Attribution
The transcription script (`transcribe_dir.sh`) is adapted from open-source examples of Whisper CLI usage. It has been modified and extended for parallel batch processing on Jetstream2 A100 GPUs. We release this version under the MIT License.
