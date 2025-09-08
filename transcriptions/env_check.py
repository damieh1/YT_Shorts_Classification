python - << 'PY'
import torch, sys, pkgutil
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("Whisper:", pkgutil.find_loader("whisper"))
PY
