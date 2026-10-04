"""Static (AST) checks on candidate source. Pure Python, no torch import.

Policy is allowlist-first:
  * imports: only modules in ALLOWED_IMPORTS (task may override);
  * module attribute chains (torch.x.y, triton.x): only allowlisted paths;
  * module objects may only appear as the base of an attribute chain (no aliasing
    `m = torch`, no passing modules around), so the allowlist cannot be bypassed;
  * no dunder access, no reflective builtins (getattr/exec/eval/open/...);
  * no assignment to attributes of modules (monkeypatching timers / torch ops);
  * host code (outside @triton.jit functions) may not call tensor compute methods
    (x.sum(), x.softmax(), ...) or use the @ operator.
Host-side arithmetic like `x + y` cannot be typed statically; the runtime launch check
(share of GPU time inside the candidate's Triton kernels) covers it.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from kernelrl.common.schemas import Finding

ENTRYPOINT = "run"

ALLOWED_IMPORTS = ("torch", "triton", "triton.language", "math", "typing")
# Modules whose every attribute is allowed (prefix match on dotted path).
OPEN_MODULES = ("triton.language", "math", "typing")

TORCH_ALLOWED = frozenset({
    "torch.empty", "torch.empty_like", "torch.empty_strided", "torch.zeros", "torch.zeros_like",
    "torch.ones", "torch.ones_like", "torch.full", "torch.full_like",
    "torch.float16", "torch.float32", "torch.float64", "torch.bfloat16", "torch.half", "torch.float",
    "torch.double", "torch.int8", "torch.int16", "torch.int32", "torch.int64", "torch.uint8", "torch.long",
    "torch.int", "torch.bool",
    "torch.Tensor", "torch.Size", "torch.dtype", "torch.device", "torch.finfo", "torch.iinfo", "torch.is_tensor",
    "torch.cuda.current_device", "torch.cuda.get_device_properties", "torch.cuda.get_device_capability",
})
TRITON_ALLOWED = frozenset({
    "triton.jit", "triton.autotune", "triton.Config", "triton.heuristics", "triton.cdiv",
    "triton.next_power_of_2", "triton.language",
})
KERNEL_DECORATORS = frozenset({"triton.jit", "triton.autotune", "triton.heuristics"})

FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "breakpoint", "input", "help", "exit", "quit", "memoryview", "__builtins__", "__loader__",
    "__spec__", "__file__",
})
_COMPUTE_METHODS = {
    "abs", "absolute", "add", "addmm", "addmv", "addcmul", "addcdiv", "amax", "amin", "aminmax", "argmax",
    "argmin", "baddbmm", "bmm", "clamp", "clamp_min", "clamp_max", "clip", "cos", "cumsum", "cumprod", "cummax",
    "cummin", "div", "divide", "dot", "einsum", "erf", "exp", "exp2", "expm1", "gelu", "log", "log1p", "log2",
    "log_softmax", "logsumexp", "logcumsumexp", "matmul", "max", "maximum", "mean", "min", "minimum", "mm", "mul",
    "multiply", "mv", "neg", "negative", "norm", "pow", "prod", "reciprocal", "relu", "rsqrt", "sigmoid",
    "silu", "sin", "softmax", "sqrt", "square", "std", "sub", "subtract", "sum", "nansum", "tanh", "var",
    "where", "masked_fill", "tril", "triu", "sort", "topk", "cross_entropy", "layer_norm", "rms_norm",
    "linear", "scaled_dot_product_attention", "normalize", "sign", "floor", "ceil", "round", "lerp", "outer",
    "kron", "tensordot", "rsub", "fmod", "remainder", "gather", "scatter", "scatter_add", "index_select",
    "take", "count_nonzero", "median", "mode", "quantile", "softplus", "hardtanh", "mish", "elu", "leaky_relu",
    "glu", "dropout", "cpu", "numpy", "tolist", "sigmoid", "logit", "hypot", "atan2", "nan_to_num",
    "diff", "trace", "diag", "inner", "chain_matmul", "conv1d", "conv2d", "unfold", "renorm", "logaddexp",
}
FORBIDDEN_METHODS = frozenset(_COMPUTE_METHODS | {m + "_" for m in _COMPUTE_METHODS})
FORBIDDEN_ATTR_STORES = frozenset({"data", "grad"})


@dataclass
class StaticResult:
    ok: bool
    reason: str = ""
    message: str = ""
    lineno: int | None = None
    syntax_error: bool = False
    findings: list[Finding] = field(default_factory=list)


class _Checker(ast.NodeVisitor):
    def __init__(self, allowed_imports: tuple[str, ...]):
        self.allowed_imports = allowed_imports
        self.mod_alias: dict[str, str] = {}  # local name -> canonical module path
        self.obj_alias: dict[str, str] = {}  # local name -> canonical attribute path (from-imports)
        self.findings: list[tuple[int, str, str]] = []
        self.in_kernel = 0
        self.kernel_count = 0

    # ---------- helpers ----------
    def flag(self, node: ast.AST, reason: str, msg: str) -> None:
        self.findings.append((getattr(node, "lineno", 0), reason, msg))

    def _import_allowed(self, mod: str) -> bool:
        return any(mod == a or mod.startswith(a + ".") and a in OPEN_MODULES for a in self.allowed_imports)

    def path_of(self, node: ast.AST) -> str | None:
        """Canonical dotted path for Name/Attribute chains rooted at an imported name."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        root = self.mod_alias.get(node.id) or self.obj_alias.get(node.id)
        if root is None:
            return None
        return ".".join([root, *reversed(parts)])

    @staticmethod
    def path_allowed(path: str) -> bool:
        if path in TORCH_ALLOWED or path in TRITON_ALLOWED:
            return True
        if any(path == m or path.startswith(m + ".") for m in OPEN_MODULES):
            return True
        return path in ("torch", "triton")  # bare module name used as attribute base only

    def _decorator_path(self, dec: ast.AST) -> str | None:
        return self.path_of(dec.func if isinstance(dec, ast.Call) else dec)

    # ---------- imports ----------
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if not self._import_allowed(a.name):
                self.flag(node, "forbidden_import", f"import of '{a.name}' is not allowed")
                continue
            if a.asname:
                self.mod_alias[a.asname] = a.name
            else:
                root = a.name.split(".")[0]
                self.mod_alias[root] = root

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level:
            self.flag(node, "forbidden_import", "relative imports are not allowed")
            return
        if not self._import_allowed(mod) and mod not in ("torch", "triton"):
            self.flag(node, "forbidden_import", f"import from '{mod}' is not allowed")
            return
        for a in node.names:
            if a.name == "*":
                self.flag(node, "forbidden_import", "star imports are not allowed")
                continue
            path = f"{mod}.{a.name}"
            local = a.asname or a.name
            if self._import_allowed(path):
                self.mod_alias[local] = path
            elif self.path_allowed(path):
                self.obj_alias[local] = path
            else:
                self.flag(node, "forbidden_api", f"'from {mod} import {a.name}' is not allowed")

    # ---------- names / attributes ----------
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.flag(node, "forbidden_builtin", f"use of '{node.id}' is not allowed")
        elif node.id.startswith("__") and node.id.endswith("__") and node.id != "__name__":
            self.flag(node, "dunder_access", f"use of '{node.id}' is not allowed")
        elif node.id in self.mod_alias:
            parent = getattr(node, "_parent", None)
            if not (isinstance(parent, ast.Attribute) and parent.value is node):
                self.flag(node, "module_escape", f"module '{node.id}' may only be used as 'module.attr'")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.flag(node, "dunder_access", f"access to '{node.attr}' is not allowed")
        parent = getattr(node, "_parent", None)
        is_chain_top = not (isinstance(parent, ast.Attribute) and parent.value is node)
        if is_chain_top:
            path = self.path_of(node)
            if path is not None and not self.path_allowed(path):
                self.flag(node, "forbidden_api", f"'{path}' is not allowed; write the computation as a Triton kernel")
            elif path in ("torch", "triton", "triton.language") and not self._as_base(node):
                self.flag(node, "module_escape", f"module '{path}' may only be used as 'module.attr'")
        self.generic_visit(node)

    def _as_base(self, node: ast.AST) -> bool:
        parent = getattr(node, "_parent", None)
        return isinstance(parent, ast.Attribute) and parent.value is node

    # ---------- statements ----------
    def _check_store_target(self, t: ast.AST) -> None:
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._check_store_target(e)
        elif isinstance(t, ast.Starred):
            self._check_store_target(t.value)
        elif isinstance(t, ast.Attribute):
            if self.path_of(t) is not None:
                self.flag(t, "module_attribute_assignment", "assigning to attributes of imported modules is not allowed")
            elif t.attr in FORBIDDEN_ATTR_STORES:
                self.flag(t, "forbidden_attribute_assignment", f"assigning to '.{t.attr}' is not allowed")
        elif isinstance(t, ast.Subscript):
            if self.path_of(t.value) is not None:
                self.flag(t, "module_attribute_assignment", "mutating imported module objects is not allowed")

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._check_store_target(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_store_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_store_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for t in node.targets:
            self._check_store_target(t)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.flag(node, "global_statement", "'global' is not allowed (no state across calls)")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.flag(node, "global_statement", "'nonlocal' is not allowed")

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        decs = {self._decorator_path(d) for d in node.decorator_list}
        is_kernel = bool(decs & KERNEL_DECORATORS)  # device-code context
        for d in node.decorator_list:
            self.visit(d)
        if "triton.jit" in decs:
            self.kernel_count += 1
        if is_kernel:
            self.in_kernel += 1
        for child in [node.args, *node.body, *([node.returns] if node.returns else [])]:
            self.visit(child)
        if is_kernel:
            self.in_kernel -= 1

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    # ---------- host-code compute ----------
    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if (not self.in_kernel and isinstance(f, ast.Attribute) and f.attr in FORBIDDEN_METHODS
                and self.path_of(f) is None):
            self.flag(node, "forbidden_tensor_method",
                      f"host code may not call '.{f.attr}()'; do the computation inside a Triton kernel")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not self.in_kernel and isinstance(node.op, ast.MatMult):
            self.flag(node, "forbidden_operator", "host code may not use '@'; write a Triton matmul kernel")
        self.generic_visit(node)


def _entrypoint(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == ENTRYPOINT:
            return node
    return None


def check_source(source: str, *, expected_arity: int | None = None,
                 allowed_imports: tuple[str, ...] | None = None, max_bytes: int = 65536) -> StaticResult:
    if len(source.encode()) > max_bytes:
        return StaticResult(False, "source_too_large", f"source exceeds {max_bytes} bytes")
    if "\x00" in source:
        return StaticResult(False, "invalid_source", "source contains NUL bytes")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return StaticResult(False, "syntax_error", f"SyntaxError: {e.msg} (line {e.lineno})", e.lineno, True)

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]

    ck = _Checker(tuple(allowed_imports or ALLOWED_IMPORTS))
    # Imports first so aliases are known regardless of statement order.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            ck.visit_Import(node)
        elif isinstance(node, ast.ImportFrom):
            ck.visit_ImportFrom(node)
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            ck.visit(node)

    entry = _entrypoint(tree)
    if entry is None:
        ck.flag(tree, "missing_entrypoint", f"no top-level 'def {ENTRYPOINT}(...)' found")
    elif expected_arity is not None:
        a = entry.args
        n_pos = len(a.posonlyargs) + len(a.args)
        n_req = n_pos - len(a.defaults)
        if a.vararg is None and not (n_req <= expected_arity <= n_pos):
            ck.flag(entry, "entrypoint_signature",
                    f"'{ENTRYPOINT}' must accept {expected_arity} positional tensor arguments")
    if ck.kernel_count == 0:
        ck.flag(tree, "no_triton_kernel", "no @triton.jit kernel found; the computation must be a Triton kernel")

    # Specific line-level violations first; file-level ones (line 0) last.
    ordered = sorted(ck.findings, key=lambda x: (x[0] == 0, x[0]))
    findings = [Finding(check=r, message=f"line {ln}: {m}" if ln else m) for ln, r, m in ordered]
    if findings:
        ln, r, m = ordered[0]
        return StaticResult(False, r, findings[0].message, ln or None, False, findings)
    return StaticResult(True)


def reference_arity(reference_source: str) -> int:
    tree = ast.parse(reference_source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "reference":
            return len(node.args.args)
    raise ValueError("reference.py has no top-level reference()")
