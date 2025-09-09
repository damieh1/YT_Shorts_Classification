# transcribe_dir.sh - batch transcription wrapper for Whisper

# Code below is adapted from community examples of Whisper CLI usage.
# Copyright (c) 2025 Daniel Miehling
# Licensed under the MIT License (see LICENSE file for details).

---

### **transcribe_dir.sh**
```bash

set -euo pipefail

# Whisper batch transcription (Linux, venv)
# USAGE:
#   ./transcribe_dir.sh INPUT_DIR OUTPUT_DIR PARALLEL_JOBS [MODEL] [LANG]

INPUT_DIR=${1:-"./videos"}
OUTPUT_DIR=${2:-"./captions"}
PARALLEL_JOBS=${3:-4}
MODEL=${4:-"large-v3"}
LANGUAGE=${5:-""}

mkdir -p "$OUTPUT_DIR"

WHISPER_ARGS=(--model "$MODEL" --device cuda --fp16 True --temperature 0 --verbose False --output_format all)

if [[ -n "$LANGUAGE" ]]; then
  WHISPER_ARGS+=(--language "$LANGUAGE")
fi

mapfile -t FILES < <(find "$INPUT_DIR" -type f \( \
  -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.m4v" -o -iname "*.webm" -o -iname "*.avi" \
\) | sort)

TOTAL=${#FILES[@]}
echo "Found $TOTAL media files under: $INPUT_DIR"

if (( TOTAL == 0 )); then
  echo "No video files found. Exiting."
  exit 0
fi

transcribe_one() {
  local f="$1"
  local base="$(basename "$f")"
  local stem="${base%.*}"

  if [[ -f "$OUTPUT_DIR/${stem}.txt" || -f "$OUTPUT_DIR/${stem}.srt" ]]; then
    echo "Skip (exists): $base"
    return 0
  fi

  echo "→ Transcribing: $base"
  whisper "$f" "${WHISPER_ARGS[@]}" --output_dir "$OUTPUT_DIR" || {
    echo "ERROR: whisper failed for $base" >&2
    return 1
  }
}

export -f transcribe_one
export OUTPUT_DIR WHISPER_ARGS

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

printf '%s\0' "${FILES[@]}" | xargs -0 -n1 -P "$PARALLEL_JOBS" bash -lc 'transcribe_one "$@"' _ || true

echo "Done. Outputs in: $OUTPUT_DIR"
COMPLETED=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.srt" | wc -l | tr -d " ")
echo "Summary: completed (srt) = $COMPLETED / $TOTAL"
