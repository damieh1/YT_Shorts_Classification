# Paraling Tier-1 — AST Domain Adaptation (News Shorts)

End-to-end pipeline to (1) score full videos, (2) curate clips for labeling in Label Studio, (3) reconcile LS JSON with WAV clips, (4) build window datasets, and (5) fine-tune AST on domain-specific events.

### Lightweight pipeline to (1) collect candidates, (2) label short audio spans in Label Studio, (3) reconcile LS JSON with actual clips, (4) build train/val window datasets, and (5) fine-tune AST (Audio Spectrogram Transformer) on domain-specific news shorts.

## TL;DR (current run)
- **Clips labeled**: 238 LS clips (from 225+ tasks)
- **Labels**: `Speech, Shout_Scream, Siren, Chant, Music_BG, Music_FG, Crowd_noise`
- **Resolved LS → audio**: 239/239 (mirror: `data/ls_resolved/`)
- **Windows** (win=0.20s, ov=0.05 for the big set; 0.25/0.10 also OK):
  - Train rows: **19,596**
  - Val rows: **4,865**
- **Fine-tune**: head-only, class-weighted BCE, padding=10s, batch=8, 8 epochs, CUDA.


## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade "transformers>=4.40" torch torchaudio soundfile numpy
# for CUDA wheels if needed:
# pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
