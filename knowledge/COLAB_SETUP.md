
# Colab Setup (Copy-Paste Cells)

This guide mirrors the live notebook `Postura_Live_Demo.ipynb` and is safe to re-run. It lets anyone open the project in a clean Colab runtime, without needing access to your Drive.

Assumptions:

- Google Colab runtime (Python 3.10/3.11 both work with the pinned stack).
- Public GitHub access to this repo.

## 1) Optional: GPU runtime

- Runtime → Change runtime type → Hardware accelerator: GPU (optional; not required for current CPU BlazePose path)

## 2) (Optional) Mount Google Drive for persistent reports

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## 3) Clone the repo (shareable for anyone)

```bash
cd /content
rm -rf Postura
git clone https://github.com/andykofman/Postura.git
cd Postura
```

## 4) Remove Colab extras that can force protobuf>=5

```bash
pip uninstall -y ydf grpcio-status || true
```

## 5) Install pinned dependencies (deterministic)

```bash
python -V
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

## 6) Hard-restart runtime so imports use the final environment

```python
import os, sys
os.kill(os.getpid(), 9)
```

## 7) After restart: re-enter the repo and (optionally) persist reports to Drive

```bash
cd /content/Postura

# Optional: make report/ persist in Drive
if [ -d /content/drive/MyDrive ]; then
  mkdir -p /content/drive/MyDrive/Postura-Reports
  rm -rf report
  ln -s /content/drive/MyDrive/Postura-Reports report
fi
```

## 8) Sanity checks (idempotent)

```bash
python scripts/colab_bootstrap.py --remove-colab-conflicts --run-mp-smoke
```

## 9) Start the API server in the background

```bash
pkill -f uvicorn || true
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 > uvicorn.log 2>&1 &
sleep 2 && sed -n '1,60p' uvicorn.log
```

## 10) Public URL via ngrok (authenticated)

- Install and authenticate ngrok, then open a tunnel to port 8000. Open the printed URL and append `/ui`.

```python
%pip -q install pyngrok
from pyngrok import ngrok
ngrok.kill()  # ensure clean slate
ngrok.set_auth_token("31Ma6WurjHpwLKZYddslQ16M8v5_7QcAgcGLyeZty1bZ7eupf")
tunnel = ngrok.connect(8000, "http", bind_tls=True)
public_url = tunnel.public_url
public_url, f"{public_url}/ui"
```

Alternative (no ngrok): use Colab’s proxy

```python
from google.colab import output
public_url = output.eval_js("google.colab.kernel.proxyPort(8000)")
public_url, f"{public_url}/ui"
```

## 11) Use the Web UI

- Open the `/ui` link. Upload an MP4 (≤ 50MB). The API saves a JSON at `report/{video_id}/summary.json`.
- If you created the Drive symlink above, this will persist under `/content/drive/MyDrive/Postura-Reports/{video_id}/summary.json`.

## 12) Troubleshooting

- UI stuck at “Uploading and analyzing...” → Check server logs:
```bash
sed -n '1,120p' uvicorn.log
```
- “NO_KNEE_SIGNAL/NO_ELBOW_SIGNAL” → ensure MediaPipe is installed (steps 4–6 succeeded). For real videos, consider relaxing confidence in `analysis/analyzer.py`:
  - `EmaSmoother(num_landmarks=33, alpha=0.21, min_confidence=0.2)`
- ngrok: “limited to 1 session” → terminate other tunnels in dashboard or run `ngrok.kill()` and retry.

## 13) Stop the server and tunnel when done

```python
from pyngrok import ngrok
ngrok.kill()
```
```bash
pkill -f uvicorn || true
```

Notes:
- This document is the canonical, copy-paste setup. Keep `Postura_Live_Demo.ipynb` for a ready-to-run, shareable notebook version.
