
# Colab Setup (Copy-Paste Cells)

Assumptions:

- Google Colab (Python 3.10).
- Replace `<REPO_URL>` with your repository URL.

## 1) Optional: GPU runtime

- Runtime → Change runtime type → Hardware accelerator: GPU (optional)

## 2) Mount Google Drive (optional, for persistence)

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## 3) Clone repo

```bash
REPO_URL="https://github.com/<YOUR_GITHUB_USER_OR_ORG>/Postura.git"
cd /content
rm -rf Postura
git clone "$REPO_URL"
cd Postura
```

## 4) System packages (idempotent)

```bash
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg
```

## 5) Remove Colab extras that can force protobuf>=5

```bash
pip uninstall -y ydf grpcio-status || true
```

## 6) Python deps with strict constraints (idempotent)

```bash
python -V
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

## 7) Hard-restart runtime so imports see the final state

```python
import os, sys
os.kill(os.getpid(), 9)
```

## 8) After restart: sanity checks

```bash
cd /content/Postura
python scripts/colab_bootstrap.py --run-mp-smoke
```

## 9) Optional: run tests

- Unit tests:

```bash
pytest -m "not integration" -q
```

- Integration test:

```bash
pytest -m integration -q
```

## Notes

- Steps are safe to re-run.
- If imports fail due to `protobuf>=5`, re-run steps 5–7.
