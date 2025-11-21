# Compute Resources

## Hardware Used
- Jetstream2 (A100 40GB, T4)
- Delta (A100 40GB)
- Local CPU cluster

## Estimated Compute Requirements
| Stage | Hardware | Estimated GPU Hours | Notes |
|-------|----------|----------------------|-------|
| Whisper Transcription | T4 | ~10 | Varies by video duration |
| Dependency Parsing | A100 | ~6–8 | SUPAR biaffine parsing ~1.5M tokens |
| ABSA Fine-Tuning | A100 | ~20–22 | Includes 5 models configurations for baseline = fine-tuning |
| ABSA Inference | A100 | ~< 1 | Predicting ~3.2k rows |
| Event Extraction | A100 | ~1 | SUPAR biaffine parsing + dependency rules |
| Computer Vision | GPU | ~25 | QWEN3 Prompt-based scene-type classification |

- All compute was performed on NSF-funded HPC clusters (Jetstream2 and Delta). Whisper transcription required approximately 10 GPU-hours on NVIDIA T4 GPUs.
- Dependency parsing using the Biaffine-SUPAR model required 6–8 GPU-hours on A100 hardware.
- Aspect-Based Sentiment Analysis (ABSA) training and hyperparameter tuning required an additional 20—22 GPU-hours on A100 GPUs. Across all baseline and tuning configurations (RoBERTa, DeBERTa-v3-base, DeBERTa-v3-large, DeBERTa-v3-large-absa-v1.1 and Qwen2.5-7B QLoRA), model training required approximately 20–22 GPU-hours on A100 GPUs.
- ABSA inference and downstream event extraction required <1 GPU-hour and <1 CPU-hour, respectively.

