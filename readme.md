# YT Shorts Project

This repository contains the full codebase for our ICWSM submission on event-level analysis of YouTube Shorts from four international news outlets.


## Repository Structure

- [`transcriptions/`](transcriptions/)  
  Whisper transcription wrapper + environment checks

- [`parsing/`](parse/)  
  Biaffine dependency parsing (SuPar)

- [`absa/`](absa/)  
  ABSA model training + inference (BERT, RoBERTa, DeBERTa, Qwen-QLoRA)

- [`CV/`](CV/)  
  Scene-type taxonomy + computer-vision implementation

- [`docs/`](docs/)  
  compute resources, dataset documententation, known artifacts & biases

- [`synthetic_data/`](synthetic_data/)  
  metadata, parses, ABSA predictions, and vision examples


## Structure

`CV/`
- Taxonomy.md # Scene-type taxonomy for Computer Vision classification

`absa/`
- `README.md` # ABSA explanation and model instructions and further instructions `requirements.txt` for each model
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

`parse/`
- `readme.md`
- `parse_whisper_captions_biaffine.py` # Biaffine dep-en, GPU usage
- `entity_dict_18_04_2025.xlsx` # Aspect categories and aspects
- `requirements.txt`

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
