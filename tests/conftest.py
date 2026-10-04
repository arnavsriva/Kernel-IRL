import shutil
import subprocess
import sys

import pytest


def _gpu_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    code = "import torch, triton; assert torch.cuda.is_available()"
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


GPU = _gpu_available()


def pytest_collection_modifyitems(config, items):
    if GPU:
        return
    skip = pytest.mark.skip(reason="needs an NVIDIA GPU with torch+triton")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
