#!/usr/bin/env python3
"""Check the installation before running anything else.

    python verify_install.py

Reports what is present, what is missing, and what each absence costs you.
Exits 0 if the platform can run, 1 if something essential is missing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED = [
    ("numpy", "numerical arrays"),
    ("pandas", "tables"),
    ("scipy", "signal processing and statistics"),
    ("sklearn", "baseline models and metrics"),
    ("yaml", "configuration files"),
    ("matplotlib", "figures"),
]
OPTIONAL = [
    ("torch", "the tabular Transformer", "python -m pip install torch"),
    ("torchvision", "pretrained image backbones", "python -m pip install torchvision"),
    ("PIL", "image loading", "python -m pip install pillow"),
    ("streamlit", "the web interface", "python -m pip install streamlit"),
    ("openpyxl", "Excel files", "python -m pip install openpyxl"),
    ("pyarrow", "Parquet files", "python -m pip install pyarrow"),
    ("pytest", "the test suite", "python -m pip install pytest"),
]
# Packages that must import with only the required dependencies present.
PACKAGES = [
    "src", "src.data", "src.evaluation", "src.feature_selection",
    "src.leakage", "src.preprocessing", "src.reporting", "src.utils",
    "src.explainability", "src.tuning",
]
# Packages that legitimately need PyTorch. Checked only when torch is present,
# so a working torch-free install is not reported as broken.
TORCH_PACKAGES = [
    "src.models.transformer", "src.models.trainer", "src.models.encoders",
    "src.models.fusion", "src.models.multimodal",
]


def main() -> int:
    print("=" * 66)
    print("QSQ-FS ML PLATFORM — INSTALLATION CHECK")
    print("=" * 66)
    print(f"\nPython {sys.version.split()[0]}")
    print(f"Running from {Path.cwd()}")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"Virtual environment: {'ACTIVE' if in_venv else 'NOT ACTIVE'}")
    if not in_venv:
        print("  ! Not in a virtual environment. On Windows, run:")
        print("      .venv\\Scripts\\activate")

    print("\nREQUIRED PACKAGES")
    print("-" * 66)
    missing = []
    for module, purpose in REQUIRED:
        try:
            loaded = importlib.import_module(module)
            version = getattr(loaded, "__version__", "?")
            print(f"  OK       {module:<14} {version:<12} {purpose}")
        except ImportError:
            print(f"  MISSING  {module:<14} {'':<12} {purpose}")
            missing.append(module)

    print("\nOPTIONAL PACKAGES")
    print("-" * 66)
    for module, purpose, install in OPTIONAL:
        try:
            loaded = importlib.import_module(module)
            version = getattr(loaded, "__version__", "?")
            print(f"  OK       {module:<14} {version:<12} {purpose}")
        except ImportError:
            print(f"  absent   {module:<14} {'':<12} {purpose}")
            print(f"           -> {install}")

    print("\nPROJECT PACKAGES")
    print("-" * 66)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    broken = []

    try:
        import torch  # noqa: F401
        to_check = PACKAGES + TORCH_PACKAGES
    except ImportError:
        to_check = PACKAGES
        print("  (PyTorch absent — skipping the packages that need it)")

    for package in to_check:
        try:
            importlib.import_module(package)
            print(f"  OK       {package}")
        except Exception as exc:
            print(f"  BROKEN   {package}  ({type(exc).__name__}: {exc})")
            broken.append(package)

    print("\nKEY FILES")
    print("-" * 66)
    root = Path(__file__).resolve().parent
    for name in ("app.py", "run_pipeline.py", "requirements.txt",
                 "config/default_config.yaml", "src/data/loader.py",
                 "src/pipeline.py"):
        exists = (root / name).exists()
        print(f"  {'OK      ' if exists else 'MISSING '} {name}")
        if not exists:
            missing.append(name)

    print("\n" + "=" * 66)
    if missing or broken:
        print("NOT READY")
        if missing:
            print(f"  Missing: {', '.join(missing)}")
            print("  Fix with: python -m pip install -r requirements.txt")
        if broken:
            print(f"  Broken imports: {', '.join(broken)}")
            print("  Usually means files are missing. Re-download the project.")
        return 1

    try:
        import torch  # noqa: F401
        print("READY — full installation, including the Transformer.")
    except ImportError:
        print("READY — the Transformer will be skipped (PyTorch not installed).")
        print("        Everything else runs: feature selection, all baselines,")
        print("        tuning, figures and statistical tables.")
        print("        To add it: python -m pip install torch")
    print("\nNext:  python run_pipeline.py --help")
    print("       streamlit run app.py")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
