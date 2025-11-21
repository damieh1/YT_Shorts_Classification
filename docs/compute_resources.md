# Compute Resources

## Hardware Used
- Jetstream2 (A100 40GB, T4)
- Delta (A100 40GB)
- Local CPU cluster

## Estimated Compute Requirements
| Stage | Hardware | Estimated GPU Hours | Notes |
|-------|----------|----------------------|-------|
| Whisper Transcription | T4 | ~XX | Varies by video duration |
| Dependency Parsing | A100 | ~XX | Parsing ~1.5M tokens |
| ABSA Fine-Tuning | A100 | ~XX | 3 epochs, batch size X |
| ABSA Inference | V100/T4 | ~XX | Predicting ~X sentences |
| Event Extraction | CPU | ~XX | Regex + dependency rules |

Exact values will be updated upon final run.
