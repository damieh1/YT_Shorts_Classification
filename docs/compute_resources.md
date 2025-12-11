# Compute Resources

## Hardware Used
- Platform: Jetstream2
- GPU: NVIDIA A100-SXM4-40GB
- Driver: 550.144.03
- CUDA: v12.4

## Compute Resources
All experiments were conducted on NSF ACCESS Jetstream2/Delta nodes equipped with NVIDIA A100-SXM4-40GB GPUs (driver version 550.144.03, CUDA 12.4). All Whisper transcription, dependency parsing, ABSA fine-tuning, ABSA inference, and event extraction were executed on this hardware class. The total compute required for transcription was approximately 10 GPU-hours. Dependency parsing required 6–8 GPU-hours. Across all ABSA model configurations (RoBERTa-base, DeBERTa-v3-base, DeBERTa-v3-large, DeBERTa-v3-large-absa-v1.1 , and Qwen2.5-7B QLoRA), fine-tuning required approximately 20–22 GPU-hours. ABSA inference and event extraction required < 1 GPU-hour and < 1 CPU-hour, respectively. Frame sampling required approximately ~5 hours, and prompt-based semantic scene classification required approximately ~495 GPU hours.

## Estimated Compute Hours
| Stage | Hardware | Estimated GPU Hours | Notes |
|-------|----------|----------------------|-------|
| Whisper Transcription | T4 | ~10 | Varies by video duration |
| Dependency Parsing | A100 | ~6–8 | SUPAR biaffine parsing ~1.5M tokens |
| ABSA Fine-Tuning | A100 | ~20–22 | Includes 5 models configurations for baseline = fine-tuning |
| ABSA Inference | A100 | ~< 1 | Predicting ~3.2k rows |
| Event Extraction | A100 | ~1 | SUPAR biaffine parsing + dependency rules |
| Frame Sampling | A100 |  ~5 | Fixed rate of one FPS across input formats |
| Computer Vision | A100 | ~490 | QWEN3 4B Prompt-based scene-type classification |


