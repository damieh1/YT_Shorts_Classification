# YT Shorts Project

This repository contains the full codebase for our full paper submission on YouTube Shorts from four international news outlets.


## Repository Structure

- `transcriptions/`
  Whisper transcription wrapper + environment checks

- `parsing/`
  Biaffine dependency parsing (SuPar)

- `absa/`
  ABSA model training + inference (BERT, RoBERTa, DeBERTa, Qwen-QLoRA)

- `CV/`
  Scene-type taxonomy + computer-vision implementation

- `data/`
  Publicly available dataset on YouTube Shorts & Snippets for GroundTruth (APC) + VLM-Model output

- `docs/`
  Compute resources, dataset documententation, known artifacts & biases

- `synthetic_data/`
  Metadata, parses, ABSA predictions
