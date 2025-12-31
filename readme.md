# YT Shorts Project

This repository contains the full codebase for our ICWSM submission on YouTube Shorts from four international news outlets.


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
  Snippet for GroundTruth (APC) for ABSA + publicly available dataset on YouTube Shorts + CV Sample Model ouput

- `docs/`
  compute resources, dataset documententation, known artifacts & biases

- `synthetic_data/`
  metadata, parses, ABSA predictions, and vision examples
