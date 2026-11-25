"""Download Stockfish into ./tools for your OS (Windows / macOS / Linux x86-64)."""
import io, os, platform, sys, tarfile, zipfile, urllib.request

BASE = "https://github.com/official-stockfish/Stockfish/releases/latest/download/"
s = platform.system()
if s == "Windows":
    asset = "stockfish-windows-x86-64-avx2.zip"
elif s == "Darwin":
    asset = "stockfish-macos-universal.tar.gz"
elif s == "Linux":
    asset = "stockfish-ubuntu-x86-64-avx2.tar"
else:
    sys.exit(f"unsupported OS {s}; install stockfish manually and set STOCKFISH_PATH")
print("downloading", asset)
data = urllib.request.urlopen(BASE + asset).read()
os.makedirs("tools", exist_ok=True)
if asset.endswith(".zip"):
    zipfile.ZipFile(io.BytesIO(data)).extractall("tools")
else:
    tarfile.open(fileobj=io.BytesIO(data)).extractall("tools")
print("extracted to ./tools  (logicalllm finds it automatically; else set STOCKFISH_PATH)")
