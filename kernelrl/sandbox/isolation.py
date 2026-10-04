"""Fresh-subprocess execution with hard timeouts and a host-memory watchdog.

The parent never imports torch, so nothing a candidate does (hang, segfault, CUDA
context corruption, OOM) can take it down. Each stage runs in its own process
group (start_new_session) and is SIGKILLed as a group on timeout. Communication
is via files in a per-job directory (no pipes, so no pipe-buffer deadlocks).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ChildOutcome:
    returncode: int | None
    timed_out: bool
    rss_exceeded: bool
    wall_s: float
    peak_rss_bytes: int
    stdout_path: Path
    stderr_path: Path

    @property
    def signal_name(self) -> str | None:
        if self.returncode is not None and self.returncode < 0:
            try:
                return signal.Signals(-self.returncode).name
            except ValueError:
                return f"signal {-self.returncode}"
        return None

    def tail(self, which: str = "stderr", nbytes: int = 4000) -> str:
        p = self.stderr_path if which == "stderr" else self.stdout_path
        try:
            with open(p, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - nbytes))
                return f.read().decode(errors="replace")
        except OSError:
            return ""


def _rss_bytes(pid: int) -> int:
    """Resident set size of a process (Linux /proc; falls back to ps elsewhere)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        return 0
    except FileNotFoundError:
        pass
    except OSError:
        return 0
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip() or 0) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def child_env(passthrough: list[str], extra: dict[str, str]) -> dict[str, str]:
    env = {k: os.environ[k] for k in passthrough if k in os.environ}
    env.update({
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONFAULTHANDLER": "1",
        "OMP_NUM_THREADS": "4",
    })
    env.update(extra)
    return env


def run_child(argv: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float,
              rss_limit_bytes: int, poll_s: float = 0.05, name: str = "child") -> ChildOutcome:
    cwd.mkdir(parents=True, exist_ok=True)
    out_p, err_p = cwd / f"{name}.stdout", cwd / f"{name}.stderr"
    t0 = time.monotonic()
    peak = 0
    timed_out = rss_exceeded = False
    is_linux = sys.platform.startswith("linux")
    rss_every = 1 if is_linux else max(1, int(0.5 / poll_s))  # ps is slow; poll it less often
    with open(out_p, "wb") as out_f, open(err_p, "wb") as err_f:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out_f, stderr=err_f,
                                start_new_session=True, close_fds=True)
        i = 0
        while True:
            try:
                proc.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            i += 1
            if time.monotonic() - t0 > timeout_s:
                timed_out = True
                _kill_group(proc)
                proc.wait()
                break
            if i % rss_every == 0:
                rss = _rss_bytes(proc.pid)
                peak = max(peak, rss)
                if rss > rss_limit_bytes:
                    rss_exceeded = True
                    _kill_group(proc)
                    proc.wait()
                    break
    # Reap any stragglers in the group (a candidate cannot fork past the static check,
    # but be defensive).
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return ChildOutcome(proc.returncode, timed_out, rss_exceeded, time.monotonic() - t0, peak, out_p, err_p)
