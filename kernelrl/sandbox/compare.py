"""Numerical comparison against the reference. Imports torch: only used inside
sandbox child processes and the task validator, never in the parent orchestrator.

Policy: the reference is computed twice:
  * natively (task dtype) to get the expected output shape and dtype;
  * on upcast inputs (fp16/bf16 -> fp32, fp32 -> fp64) to get high-precision values.
The candidate output is compared against the high-precision values with the task's
atol/rtol, elementwise: |out - ref| <= atol + rtol * |ref| (no mismatch allowance).
"""
from __future__ import annotations

import torch

UPCAST = {torch.float16: torch.float32, torch.bfloat16: torch.float32, torch.float32: torch.float64}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def upcast(inputs):
    return tuple(x.to(UPCAST[x.dtype]) if isinstance(x, torch.Tensor) and x.dtype in UPCAST else x for x in inputs)


def clone_inputs(inputs):
    return tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)


@torch.no_grad()
def reference_outputs(ref_fn, inputs):
    """Returns (expected_shape, expected_dtype, high_precision_values)."""
    native = ref_fn(*clone_inputs(inputs))
    hi = ref_fn(*upcast(clone_inputs(inputs)))
    return tuple(native.shape), native.dtype, hi.detach()


def compare(out, ref_hi, expected_shape, expected_dtype, expected_device, atol: float, rtol: float) -> dict:
    res = {"passed": False, "error": None, "max_abs_err": None, "max_rel_err": None, "mismatch_frac": None}
    if not isinstance(out, torch.Tensor):
        res["error"] = f"output is {type(out).__name__}, expected torch.Tensor"
        return res
    if tuple(out.shape) != tuple(expected_shape):
        res["error"] = f"shape_mismatch: got {tuple(out.shape)} expected {tuple(expected_shape)}"
        return res
    if out.dtype != expected_dtype:
        res["error"] = f"dtype_mismatch: got {out.dtype} expected {expected_dtype}"
        return res
    if out.device != expected_device:
        res["error"] = f"device_mismatch: got {out.device} expected {expected_device}"
        return res
    o = out.detach().to(ref_hi.dtype)
    ref_finite = torch.isfinite(ref_hi)
    bad_nonfinite = (~torch.isfinite(o)) & ref_finite
    if bool(bad_nonfinite.any()):
        n = int(bad_nonfinite.sum())
        has_nan = bool(torch.isnan(o).any())
        res["error"] = f"{'nan' if has_nan else 'inf'}_in_output: {n} non-finite values where reference is finite"
        res["mismatch_frac"] = n / max(1, o.numel())
        return res
    if o.numel() == 0:
        res.update(passed=True, max_abs_err=0.0, max_rel_err=0.0, mismatch_frac=0.0)
        return res
    # Non-finite reference positions (e.g. -inf) must match exactly.
    nf_mismatch = (~ref_finite) & ~((o == ref_hi) | (torch.isnan(o) & torch.isnan(ref_hi)))
    diff = (o - ref_hi).abs().masked_fill(~ref_finite, 0)
    rel = diff / ref_hi.abs().masked_fill(~ref_finite, 1).clamp_min(1e-30)
    bad = (diff > atol + rtol * ref_hi.abs().masked_fill(~ref_finite, 0)) | nf_mismatch
    res["max_abs_err"] = float(diff.max())
    res["max_rel_err"] = float(rel.max())
    res["mismatch_frac"] = float(bad.float().mean())
    if bool(bad.any()):
        idx = int(bad.flatten().nonzero()[0])
        res["error"] = (f"value_mismatch: {res['mismatch_frac']:.2%} of elements outside tolerance "
                        f"(atol={atol}, rtol={rtol}); max_abs_err={res['max_abs_err']:.3e}; first bad flat index {idx}: "
                        f"got {float(o.flatten()[idx]):.6g} expected {float(ref_hi.flatten()[idx]):.6g}")
        return res
    res["passed"] = True
    return res


def bitwise_equal(a, b) -> bool:
    """Exact equality including NaN payloads (NaN == NaN here)."""
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    if a.dtype.is_floating_point:
        nbits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return bool(torch.equal(a.contiguous().view(nbits), b.contiguous().view(nbits)))
    return bool(torch.equal(a, b))


def shares_storage(a, b) -> bool:
    """True if two tensors' memory ranges overlap."""
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.device != b.device:
        return False
    if a.numel() == 0 or b.numel() == 0:
        return False
    sa, sb = a.untyped_storage(), b.untyped_storage()
    a0, a1 = sa.data_ptr(), sa.data_ptr() + sa.nbytes()
    b0, b1 = sb.data_ptr(), sb.data_ptr() + sb.nbytes()
    return a0 < b1 and b0 < a1
