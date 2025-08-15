
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any, Dict, List, Tuple
import subprocess


def run(cmd: str) -> Tuple[int, str]:
	proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
	return proc.returncode, proc.stdout


def check_python_version() -> Dict[str, Any]:
	vi = sys.version_info
	ok = (vi.major, vi.minor) == (3, 10)
	return {"name": "python_version", "detected": f"{vi.major}.{vi.minor}.{vi.micro}", "expected": "3.10.x", "ok": ok}


def check_files() -> List[Dict[str, Any]]:
	required = ["requirements.txt", "constraints.txt"]
	return [{"name": f"file:{f}", "exists": os.path.exists(f)} for f in required]


def check_apt_packages() -> Dict[str, Any]:
	if platform.system().lower() != "linux":
		return {"name": "apt_packages", "ok": True, "note": "Non-Linux; apt checks skipped.", "missing": []}

	to_check = ["libgl1", "libglib2.0-0", "ffmpeg"]
	missing: List[str] = []
	for pkg in to_check:
		code, _ = run(f"dpkg -s {pkg} >/dev/null 2>&1")
		if code != 0:
			missing.append(pkg)
	ok = len(missing) == 0
	return {"name": "apt_packages", "ok": ok, "missing": missing}


def check_python_imports() -> Dict[str, Any]:
	missing: List[str] = []
	for mod in ["numpy", "cv2", "mediapipe"]:
		try:
			__import__(mod)
		except Exception:
			missing.append(mod)
	return {"name": "python_imports", "ok": len(missing) == 0, "missing": missing}


def check_protobuf_version() -> Dict[str, Any]:
	try:
		import google.protobuf as _pb  # type: ignore
		ver = getattr(_pb, "__version__", "unknown")
		return {"name": "protobuf_version", "version": ver, "ok": ver.startswith("4.")}
	except Exception as exc:
		return {"name": "protobuf_version", "error": str(exc), "ok": False}


def posebackend_sanity() -> Dict[str, Any]:
	try:
		import numpy as np  # type: ignore
		from pose.backend import PoseBackend
		class _FakePose:
			def process(self, frame_rgb):
				class _R:
					pose_landmarks = None
				return _R()
		with PoseBackend(pose_model=_FakePose()) as backend:
			frame = np.zeros((32, 32, 3), dtype=np.uint8)
			kps = backend.infer(frame)
			ok = isinstance(kps, list) and len(kps) == 33
		return {"name": "posebackend_sanity", "ok": ok}
	except Exception as exc:
		return {"name": "posebackend_sanity", "ok": False, "error": str(exc)}


def main() -> None:
	results: List[Dict[str, Any]] = []
	results.append(check_python_version())
	results.extend(check_files())
	results.append(check_apt_packages())
	results.append(check_python_imports())
	results.append(check_protobuf_version())
	results.append(posebackend_sanity())

	all_ok = all(item.get("ok", True) for item in results if "ok" in item)

	print(json.dumps({"results": results, "all_ok": all_ok, "environment": platform.platform()}, indent=2))

	if all_ok:
		print("PRE-DOCKER CHECK: PASS")
		print("Notes:")
		print("- Inside Docker we will preinstall apt packages and pinned wheels.")
		print("- CPU base image first; GPU optional via NVIDIA toolkit.")
	else:
		print("PRE-DOCKER CHECK: FAIL")
		sys.exit(1)


if __name__ == "__main__":
	main()
