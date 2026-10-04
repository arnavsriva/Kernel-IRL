"""The parent survives hostile children (no GPU needed)."""
import sys

import pytest

from kernelrl.sandbox.isolation import child_env, run_child

ENV = child_env(["PATH", "HOME"], {})


def _run(tmp_path, code, **kw):
    kw.setdefault("timeout_s", 5)
    kw.setdefault("rss_limit_bytes", 1 << 30)
    return run_child([sys.executable, "-c", code], cwd=tmp_path, env=ENV, name="t", **kw)


def test_ok(tmp_path):
    o = _run(tmp_path, "print('hello')")
    assert o.returncode == 0 and not o.timed_out and "hello" in o.tail("stdout")


def test_timeout_kills(tmp_path):
    o = _run(tmp_path, "while True: pass", timeout_s=1)
    assert o.timed_out and o.signal_name == "SIGKILL" and o.wall_s < 4


def test_timeout_kills_process_group(tmp_path):
    code = ("import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "time.sleep(60)")
    o = _run(tmp_path, code, timeout_s=1)
    assert o.timed_out


def test_segfault(tmp_path):
    o = _run(tmp_path, "import ctypes; ctypes.string_at(0)")
    assert o.signal_name in ("SIGSEGV", "SIGBUS")


def test_exception_exit_code(tmp_path):
    o = _run(tmp_path, "raise RuntimeError('boom')")
    assert o.returncode == 1 and "boom" in o.tail()


def test_memory_watchdog(tmp_path):
    o = _run(tmp_path, "import time\nx = b'\\x01' * (3 * 1024**3)\ntime.sleep(30)", timeout_s=20,
             rss_limit_bytes=1 << 30)
    assert o.rss_exceeded and o.wall_s < 15


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits")
def test_file_size_limit(tmp_path):
    code = ("import resource\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))\n"
            "while True: print('x' * 100000, flush=True)")
    o = _run(tmp_path, code)
    # CPython ignores SIGXFSZ, so the write raises OSError(EFBIG) instead of killing the process;
    # the sandbox child classifies that as crash/file_size_limit.
    assert o.signal_name == "SIGXFSZ" or "File too large" in o.tail()
    assert (tmp_path / "t.stdout").stat().st_size <= 1 << 20
