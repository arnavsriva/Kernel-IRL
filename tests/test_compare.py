import torch

from kernelrl.sandbox.compare import compare


def test_nan_message_points_at_unwritten_elements():
    ref = torch.zeros(4, 4, dtype=torch.float64)
    out = torch.zeros(4, 4)
    out[2:] = float("nan")
    r = compare(out, ref, (4, 4), torch.float32, out.device, atol=1e-5, rtol=1e-5)
    assert not r["passed"] and r["error"].startswith("nan_in_output: 8 of 16")
    assert "never wrote" in r["error"] and "launch grid" in r["error"]


def test_shape_dtype_and_values():
    ref = torch.ones(3, dtype=torch.float64)
    assert compare(torch.ones(3), ref, (3,), torch.float32, torch.device("cpu"), atol=0, rtol=0)["passed"]
    assert "shape_mismatch" in compare(torch.ones(4), ref, (3,), torch.float32, torch.device("cpu"), atol=0, rtol=0)["error"]
    assert "dtype_mismatch" in compare(torch.ones(3).half(), ref, (3,), torch.float32, torch.device("cpu"), atol=0, rtol=0)["error"]
    r = compare(torch.tensor([1.0, 1.0, 1.5]), ref, (3,), torch.float32, torch.device("cpu"), atol=1e-3, rtol=1e-3)
    assert "value_mismatch" in r["error"] and r["mismatch_frac"] > 0.3
