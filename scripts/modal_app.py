"""Run the Phase 0 GPU steps on Modal.

GPU type is read at launch (default "H100!" = no silent upgrade to H200, so timings are comparable):
  modal run --detach scripts/modal_app.py --step probe                        # ~5-10 GPU-minutes, do this first
  KERNELRL_GPU=L40S modal run --detach scripts/modal_app.py --step verdicts   # cheaper GPU for debugging
  modal run --detach scripts/modal_app.py --step validate
  modal run --detach scripts/modal_app.py --step eval --eval-args "--k 2 --tasks l1_vector_add l2_softmax"   # smoke
  modal run --detach scripts/modal_app.py --step eval                         # full baseline (hours)
  modal run --detach scripts/modal_app.py --step bench
  modal run scripts/modal_app.py --step fetch                                 # results from the Volume, no GPU

Always pass --detach for GPU steps (modal run --detach ...): without it, a dropped connection from
your machine stops the app. If the local client disconnects, the run continues and `--step fetch`
retrieves the results afterwards.

The repo is copied into the container, so the run uses your current working tree. runs/, results,
and the baseline-timing cache live on a Modal Volume, so an interrupted eval resumes. Result files
are copied back into docs/results/ locally (KERNELRL_GPU and the GPU name are in every result's
env fingerprint).
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
GPU = os.environ.get("KERNELRL_GPU", "H100!")
IMAGE_TAG = "slimerl/sglang:v0.5.15.post1-cu129"  # the image slime builds on (see pyproject pins)
MOUNT, WORK, VOL = "/root/kernelrl_src", "/root/KernelIRL", "/vol"

image = (
    modal.Image.from_registry(IMAGE_TAG)
    .run_commands("python3 -m pip install --break-system-packages pytest==8.4.2 PyYAML==6.0.2 "
                  "|| python3 -m pip install pytest==8.4.2 PyYAML==6.0.2")
    .env({"HF_HOME": f"{VOL}/hf", "KERNELRL_CACHE_DIR": f"{VOL}/cache", "PYTHONPATH": WORK})
    .add_local_dir(REPO, MOUNT, ignore=[".venv", ".git", "runs", ".kernelrl_cache", "**/__pycache__",
                                        ".pytest_cache", "*.egg-info", "docs/results/*.json"])
)
vol = modal.Volume.from_name("kernelrl-phase0", create_if_missing=True)
app = modal.App("kernelrl-phase0", image=image)

STEP_TIMEOUT = {"probe": 1800, "diagnose": 1800, "validate": 3600, "verdicts": 3 * 3600, "eval": 10 * 3600, "bench": 2 * 3600}
LOCAL_STEPS = ("fetch",)  # no GPU: copy results/ from the Volume (use after a local disconnect)


def _prepare_workdir() -> None:
    import shutil

    shutil.rmtree(WORK, ignore_errors=True)
    shutil.copytree(MOUNT, WORK)
    for sub in ("runs", "results", "cache", "hf"):
        os.makedirs(f"{VOL}/{sub}", exist_ok=True)
    shutil.rmtree(f"{WORK}/runs", ignore_errors=True)
    os.symlink(f"{VOL}/runs", f"{WORK}/runs")
    shutil.rmtree(f"{WORK}/docs/results", ignore_errors=True)
    os.symlink(f"{VOL}/results", f"{WORK}/docs/results")


def _collect_results() -> dict[str, str]:
    out = {}
    for p in Path(f"{VOL}/results").glob("*"):
        if p.is_file() and p.stat().st_size < 50 << 20:
            out[p.name] = p.read_text(errors="replace")
    return out


@app.function(gpu=GPU, timeout=max(STEP_TIMEOUT.values()), volumes={VOL: vol}, cpu=8, memory=49152)
def run_step(step: str, eval_args: str = "", bench_args: str = "", run_id: str = "", fixtures: str = "") -> dict:
    import subprocess

    _prepare_workdir()
    env = {**os.environ, "STEPS": step, "PY": "python3", "GPU": "0", "EVAL_ARGS": eval_args,
           "BENCH_ARGS": bench_args, "FIXTURES": fixtures}
    if run_id:
        env["RUN_ID"] = run_id
    subprocess.run(["nvidia-smi"], check=False)
    try:
        proc = subprocess.run(["bash", "scripts/run_phase0_gpu.sh"], cwd=WORK, env=env)
        rc = proc.returncode
    finally:
        vol.commit()
    return {"returncode": rc, "files": _collect_results()}


def fetch_results() -> int:
    """Copy every file under results/ on the Volume into docs/results/ (no container, no GPU)."""
    dest = REPO / "docs" / "results"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry in vol.listdir("results"):
        name = Path(entry.path).name
        (dest / name).write_bytes(b"".join(vol.read_file(entry.path)))
        n += 1
    print(f"[modal] fetched {n} result files from Volume 'kernelrl-phase0' into {dest}")
    return n


@app.local_entrypoint()
def main(step: str = "probe", eval_args: str = "", bench_args: str = "", run_id: str = "", fixtures: str = ""):
    if step in LOCAL_STEPS:
        fetch_results()
        return
    if step not in STEP_TIMEOUT:
        raise SystemExit(f"unknown step {step!r}; choose from {list(STEP_TIMEOUT) + list(LOCAL_STEPS)}")
    print(f"[modal] step={step} gpu={GPU} image={IMAGE_TAG}")
    res = run_step.remote(step, eval_args, bench_args, run_id, fixtures)
    dest = REPO / "docs" / "results"
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in res["files"].items():
        (dest / name).write_text(text)
    print(f"[modal] step={step} returncode={res['returncode']}; copied {len(res['files'])} result files to {dest}")
    if res["returncode"] != 0:
        raise SystemExit(res["returncode"])
