"""Make the pip-installed CUDA runtime DLLs findable before ctranslate2 loads.

ctranslate2 links against cuBLAS and cuDNN 9 but does not ship them. On Linux the
nvidia-*-cu12 wheels are found through LD_LIBRARY_PATH; on Windows nothing puts their
`bin` directories on the DLL search path, so importing faster_whisper fails with

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

Installing torch used to mask this, because its wheels bundle the same DLLs and its own
import registered that directory. Without torch the registration has to happen here.

`register()` must run before anything imports ctranslate2, which faster_whisper does at
import time. It is called from app/__init__.py, so every import of `app.*` is covered.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

# Subdirectory holding the DLLs differs by platform: bin on Windows, lib elsewhere.
_SUBDIR = "bin" if sys.platform == "win32" else "lib"
_PACKAGES = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")

_registered = False


def _candidate_dirs() -> list[Path]:
    roots = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.append(Path(path) / "nvidia")
    found = []
    for root in dict.fromkeys(roots):  # de-duplicate, keep order
        for package in _PACKAGES:
            directory = root / package / _SUBDIR
            if directory.is_dir():
                found.append(directory)
    return found


def register() -> list[Path]:
    """Add the CUDA DLL directories to the search path. Safe to call more than once."""
    global _registered
    if _registered:
        return []

    directories = _candidate_dirs()
    for directory in directories:
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass
        # Also prepend to PATH. add_dll_directory covers dependencies resolved through
        # the default search order, but some loaders still consult PATH, and this costs
        # nothing.
        if str(directory) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")

    _registered = True
    return directories
