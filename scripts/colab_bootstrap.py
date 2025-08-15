
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import sys
from typing import Tuple


def is_colab() -> bool:
	try:
		import google.colab  # type: ignore
		return True
	except Exception:
		return False


def run(cmd: str) -> Tuple[int, str]:
	proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
	return proc.returncode, proc.stdout


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
	update_code, update_out = run("sudo apt-get update -y")
	if update_code != 0:
		print(update_out)
		raise SystemExit("[apt] apt-get update failed")

	install_cmd = "sudo apt-get install -y --no-install-recommends " + " ".join(missing)
	code, out = run(install_cmd)
	print(out)
	if code != 0:
		raise SystemExit("[apt] apt-get install failed")


def check_python_version(expected_major: int, expected_minor: int) -> None:
	vi = sys.version_info
	print(f"[python] Detected Python {vi.major}.{vi.minor}.{vi.micro}")
	if (vi.major, vi.minor) != (expected_major, expected_minor):
		print(f"[warn] Expected Python {expected_major}.{expected_minor} for best compatibility.")


def remove_colab_conflicts(packages: list[str]) -> None:
	if not is_colab():
		return
	to_remove = [p for p in packages if importlib.util.find_spec(p.replace("-", "_")) is not None]
	if not to_remove:
		print("[colab] No conflicting preinstalled packages detected.")
		return
	print(f"[colab] Removing conflicting preinstalled packages: {', '.join(to_remove)}")
	code, out = run("pip uninstall -y " + " ".join(to_remove))
	print(out)
	if code != 0:
		print("[colab] Warning: uninstall may have partially failed. Consider restarting runtime.")


def ensure_pip_packages(requirements: str = "requirements.txt", constraints: str = "constraints.txt") -> None:
	if not os.path.exists(requirements):
		raise SystemExit(f"[pip] Missing {requirements}")
	if not os.path.exists(constraints):
		raise SystemExit(f"[pip] Missing {constraints}")

	print("[pip] Installing/upgrading packages per requirements + constraints (idempotent).")
	code, out = run("python -m pip install --upgrade pip")
	print(out)
	code, out = run(f"pip install -r {requirements} -c {constraints}")
	print(out)
	if code != 0:
		raise SystemExit("[pip] pip install failed")


def verify_imports() -> None:
	print("[verify] Importing numpy, cv2, mediapipe...")
	import numpy as np  # noqa: F401
	import cv2  # type: ignore  # noqa: F401
	import mediapipe as mp  # type: ignore  # noqa: F401
	print("[verify] Imports OK.")


def check_protobuf_version() -> None:
	try:
		import google.protobuf as _pb  # type: ignore
		ver = getattr(_pb, "__version__", "unknown")
		print(f"[verify] protobuf version: {ver}")
		if not ver.startswith("4."):
			print("[warn] protobuf>=5 detected; MediaPipe may not work. Restart runtime after uninstalling conflicts.")
	except Exception:
		print("[warn] Could not determine protobuf version.")


def mediapipe_smoke_test() -> None:
	print("[mp] Running minimal BlazePose smoke test...")
	import numpy as np
	import mediapipe as mp  # type: ignore

	img = np.zeros((128, 128, 3), dtype=np.uint8)  # BGR
	with mp.solutions.pose.Pose(static_image_mode=True) as pose:
		_ = pose.process(img[..., ::-1])  # RGB
	print("[mp] BlazePose smoke test OK.")


def main() -> None:
	parser = argparse.ArgumentParser(description="Colab bootstrap for Postura (idempotent).")
	parser.add_argument("--run-mp-smoke", action="store_true", help="Run a minimal MediaPipe smoke test.")
	parser.add_argument("--remove-colab-conflicts", action="store_true", help="Uninstall Colab extras (ydf, grpcio-status).")
	parser.add_argument("--force-runtime-restart", action="store_true", help="Force-kill current runtime process (Colab).")
	args = parser.parse_args()

	if is_colab():
		print("[env] Google Colab environment detected.")
	else:
		print("[env] Non-Colab environment. Proceeding anyway.")

	check_python_version(3, 10)
	ensure_apt_packages(["libgl1", "libglib2.0-0", "ffmpeg"])

	if args.remove_colab_conflicts:
		remove_colab_conflicts(["ydf", "grpcio-status"])

	ensure_pip_packages()
	verify_imports()
	check_protobuf_version()
	if args.run-mp-smoke:
		mediapipe_smoke_test()

	if args.force_runtime_restart and is_colab():
		print("[env] Forcing runtime restart...")
		os.kill(os.getpid(), 9)

	print("[done] Colab bootstrap completed successfully.")


if __name__ == "__main__":
	main()
