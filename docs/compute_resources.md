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
| ABSA Fine-Tuning | A100 | ~10 | 3 epochs, batch size X |
| ABSA Inference | V100/T4 | ~< 1 | Predicting ~X sentences |
| Event Extraction | GPU | ~1 | SUPAR biaffine parsing + dependency rules |

