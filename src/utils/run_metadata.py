"""Capture the environment a run happened in, for reproducibility."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def get_metadata(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    for name in ("numpy", "pandas", "sklearn", "scipy", "torch"):
        try:
            module = __import__(name)
            meta[f"{name}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            meta[f"{name}_version"] = None
    try:
        import torch

        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["device_count"] = int(torch.cuda.device_count())
    except ImportError:
        meta["cuda_available"] = False
        meta["device_count"] = 0
    if extra:
        meta.update(extra)
    return meta
