#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from typing import Tuple


def run(cmd: str) -> Tuple[int, str]:
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def ensure_apt_packages(packages: list[str]) -> None:
    if platform.system().lower() != "linux":
        print("[apt] Non-Linux detected; skipping apt installation.")
        return
    missing = []
    for pkg in packages:
        code, _ = run(f"dpkg -s {pkg} >/dev/null 2>&1")
        if code != 0:
            missing.append(pkg)
    if not missing:
        print("[apt] All required packages present.")
        return
    print(f"[apt] Installing missing packages: {', '.join(missing)}")
    code, out = run("sudo apt-get update -y && sudo apt-get install -y --no-install-recommends " + " ".join(missing))
    print(out)
    if code != 0:
        raise SystemExit("[apt] apt-get install failed")


def uninstall_colab_conflicts() -> None:
    # Uninstall even if not present; pip will skip gracefully
    print("[colab] Removing potential conflicts (ydf, grpcio-status) if present...")
    code, out = run("pip uninstall -y ydf grpcio-status")
    print(out)
    if code != 0:
        print("[colab] Warning: uninstall returned non-zero. Proceeding anyway.")


def ensure_pip(requirements: str = "requirements.txt", constraints: str = "constraints.txt") -> None:
    if not os.path.exists(requirements) or not os.path.exists(constraints):
        raise SystemExit("[pip] Missing requirements.txt or constraints.txt. Run from repo root.")
    print("[pip] Upgrading pip and installing pinned dependencies...")
    _, out = run("python -m pip install --upgrade pip")
    print(out)
    code, out = run(f"pip install -r {requirements} -c {constraints}")
    print(out)
    if code != 0:
        raise SystemExit("[pip] Installation failed")


def verify_imports() -> None:
    print("[verify] Importing numpy, cv2, mediapipe...")
    import numpy as np  # noqa: F401
    import cv2  # type: ignore  # noqa: F401
    import mediapipe as mp  # type: ignore  # noqa: F401
    print("[verify] Imports OK.")


def mp_smoke() -> None:
    print("[mp] Running minimal BlazePose smoke test...")
    import numpy as np
    import mediapipe as mp  # type: ignore
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        _ = pose.process(img[..., ::-1])
    print("[mp] Smoke test OK.")


def run_tests(integration: bool = False) -> None:
    marker = "integration" if integration else "not integration"
    print(f"[tests] Running pytest -m \"{marker}\"")
    code, out = run(f"pytest -m \"{marker}\" -q")
    print(out)
    if code != 0:
        raise SystemExit("[tests] Pytest failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quickstart automation for Postura (Colab-friendly).")
    parser.add_argument("--skip-apt", action="store_true", help="Skip apt package checks/installs.")
    parser.add_argument("--skip-uninstall", action="store_true", help="Skip uninstalling Colab conflicts.")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests after setup.")
    parser.add_argument("--run-integration", action="store_true", help="Run integration tests (real mediapipe).")
    parser.add_argument("--restart", action="store_true", help="Force runtime restart at the end (Colab only).")
    args = parser.parse_args()

    print("[env] Colab detected:" , is_colab())
    if not args.skip_apt:
        ensure_apt_packages(["libgl1", "libglib2.0-0", "ffmpeg"])
    if not args.skip_uninstall:
        uninstall_colab_conflicts()
    ensure_pip()
    verify_imports()
    mp_smoke()

    if args.run_tests:
        run_tests(integration=False)
    if args.run_integration:
        run_tests(integration=True)

    if args.restart and is_colab():
        print("[env] Restarting runtime...")
        os.kill(os.getpid(), 9)

    print("[done] Quickstart completed successfully.")


if __name__ == "__main__":
    main()


