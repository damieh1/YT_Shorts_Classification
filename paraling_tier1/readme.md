# Paraling Tier-1 — AST Domain Adaptation (News Shorts)

End-to-end pipeline to (1) score full videos, (2) curate clips for labeling in Label Studio, (3) reconcile LS JSON with WAV clips, (4) build window datasets, and (5) fine-tune AST on domain-specific events.

## Labels
`Speech, Shout_Scream, Siren, Chant, Music_BG, Music_FG, Crowd_noise`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade "transformers>=4.40" torch torchaudio soundfile numpy
# for CUDA wheels if needed:
# pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
