# YT Shorts Project

This repository contains the full codebase for our ICWSM submission on YouTube Shorts from four international news outlets.


## Repository Structure

- [`transcriptions/`](https://anonymous.4open.science/r/YTSC-0E2D/transcriptions/readme.md)  
  Whisper transcription wrapper + environment checks

- [`parsing/`](https://anonymous.4open.science/r/YTSC-0E2D/parse/)  
  Biaffine dependency parsing (SuPar)

- [`absa/`](https://anonymous.4open.science/r/YTSC-0E2D/absa/)  
  ABSA model training + inference (BERT, RoBERTa, DeBERTa, Qwen-QLoRA)

- [`CV/`](https://anonymous.4open.science/r/YTSC-0E2D/CV/)  
  Scene-type taxonomy + computer-vision implementation

- [`data/`](https://anonymous.4open.science/r/YTSC-0E2D/data/)  
  Snippet for GroundTruth (APC) for ABSA + publicly available dataset on YouTube Shorts

- [`docs/`](https://anonymous.4open.science/r/YTSC-0E2D/docs/)  
  compute resources, dataset documententation, known artifacts & biases

- [`synthetic_data/`](https://anonymous.4open.science/r/YTSC-0E2D/synthetic_data/)  
  metadata, parses, ABSA predictions, and vision examples
