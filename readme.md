# YT Shorts Project

This repository contains the full codebase for our ICWSM submission on YouTube Shorts from four international news outlets.


## Repository Structure

- [`transcriptions/`](anonymous.4open.science/r/YTSC-0E2D/transcriptions/)  
  Whisper transcription wrapper + environment checks

- [`parsing/`](parse/)  
  Biaffine dependency parsing (SuPar)

- [`absa/`](absa/)  
  ABSA model training + inference (BERT, RoBERTa, DeBERTa, Qwen-QLoRA)

- [`CV/`](CV/)  
  Scene-type taxonomy + computer-vision implementation

- [`data/`](data/)  
  Snippet for GroundTruth (APC) for ABSA + publicly available dataset on YouTube Shorts

- [`docs/`](docs/)  
  compute resources, dataset documententation, known artifacts & biases

- [`synthetic_data/`](synthetic_data/)  
  metadata, parses, ABSA predictions, and vision examples
