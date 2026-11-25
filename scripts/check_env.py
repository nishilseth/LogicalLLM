"""Tell me what hardware I have and which training profile to use."""
import sys
sys.path.insert(0, ".")
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 2**30
    print(f"CUDA GPU: {p.name}, {gb:.1f} GB VRAM")
    print("recommended profile:", "gpu" if gb >= 3.5 else "mac (very low VRAM: use --profile mac)")
elif torch.backends.mps.is_available():
    print("Apple MPS GPU -> use --profile mac")
else:
    print("NO GPU FOUND. If you have an NVIDIA GPU, install the CUDA build of torch (see requirements.txt). CPU will be extremely slow.")
try:
    from logicalllm.chess.engine import find_stockfish
    print("Stockfish:", find_stockfish())
except FileNotFoundError:
    print("Stockfish: not found -> python scripts/setup_stockfish.py")
