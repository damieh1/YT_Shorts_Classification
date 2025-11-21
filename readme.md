# YT Shorts Project: Multimodal Event and Sentiment Analysis

This repository contains the full codebase for the ICWSM 2026 submission on event-level analysis of YouTube Shorts from four international news outlets.

## Structure

`CV/`
- Taxonomy.md # Scene-type taxonomy for Computer Vision classification

`absa/`
- README.md # ABSA explanation and model instructions and further instructions `requirements.txt` for each model
- `finetune_bert.py` # BERT fine-tuning
- `finetune_deberta.py` # DeBERTa-v3 fine-tuning
- `finetune_qwen_qlora.py` # Qwen2.5-7B (QLoRA) fine-tuning
- `finetune_roberta.py` # RoBERTa fine-tuning
- `predict_absa.py` # ABSA inference script

`data/`
- `Ground_truth_APC.raw` # Example ABSA training triplets (not real user data)

`docs/`
- `compute_resources.md` # GPU usage, hardware, compute hours
- `dataset_documentation.md` # Data schema and access limitations
- `known_artifacts_biases.md` # Known artifacts in transcripts + parsing
- `reproducibility_notes.md` # Full pipeline reproduction guide

`synthetic_data/`
- `example_absa_plus_parses.csv`
- `example_events.csv`
- `example_metadata.csv`
- `example_transcript.json`
- `readme.md` # Description of synthetic dataset

`transcriptions/`
- `transcribe_dir.sh` # Whisper batch scri[t
- `env_check.py` # Environment diagnostics
- `requirements.txt` # Required packs
- `readme.md`

- `.gitignore`
- `LICENSE`
- `requirements.txt`
