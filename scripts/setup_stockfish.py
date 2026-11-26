"""Download Stockfish into ./tools for your OS. Asset names are looked up from the latest release,
so this keeps working when Stockfish renames its builds."""
import io, json, os, platform, sys, tarfile, urllib.request, zipfile

API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
system, machine = platform.system(), platform.machine().lower()
arm = machine in ("arm64", "aarch64")
want = {"Windows": "windows-arm64" if arm else "windows-x86-64",
        "Darwin": "macos",
        "Linux": "linux-arm64" if arm else "linux-x86-64"}.get(system)
if not want:
    sys.exit(f"unsupported OS {system}; install Stockfish manually and set STOCKFISH_PATH")

req = lambda u: urllib.request.Request(u, headers={"User-Agent": "logicalllm-setup"})
assets = json.load(urllib.request.urlopen(req(API), timeout=60))["assets"]
cands = [a for a in assets if want in a["name"] and a["name"].endswith((".zip", ".tar.gz", ".tar"))]
cands.sort(key=lambda a: ("universal" not in a["name"], a["name"]))     # prefer the universal build
if not cands:
    sys.exit("no matching build; assets: " + ", ".join(a["name"] for a in assets))
a = cands[0]
print(f"downloading {a['name']} ({a['size'] // 10**6} MB)...")
data = urllib.request.urlopen(req(a["browser_download_url"]), timeout=300).read()
os.makedirs("tools", exist_ok=True)
if a["name"].endswith(".zip"):
    zipfile.ZipFile(io.BytesIO(data)).extractall("tools")
else:
    tarfile.open(fileobj=io.BytesIO(data), mode="r:*").extractall("tools")
print("extracted to ./tools - logicalllm finds it automatically (or set STOCKFISH_PATH)")
