from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any


def _normalize_bool_tokens(expr: str) -> str:
    expr = expr.replace(" true", " True").replace(" false", " False")
    expr = expr.replace("(true", "(True").replace("(false", "(False")
    expr = expr.replace("[true", "[True").replace("[false", "[False")
    expr = expr.replace(",true", ",True").replace(",false", ",False")
    expr = expr.replace("=true", "=True").replace("=false", "=False")
    return expr


def _parse_call(expr: str) -> ast.Call:
    expr = _normalize_bool_tokens(expr.strip())
    node = ast.parse(expr, mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Expected call expression, got: {expr!r}")
    if not isinstance(node.func, ast.Name):
        raise ValueError(f"Only simple call names are supported, got: {expr!r}")
    return node


def _const(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _const(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +v
        return -v
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_const(elt) for elt in node.elts]
    raise ValueError(f"Unsupported literal in optuna spec: {ast.dump(node)}")


def _call_kwargs(node: ast.Call) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs is not supported in optuna spec")
        out[str(kw.arg)] = _const(kw.value)
    return out


def suggest_from_string(trial, name: str, spec: str) -> Any:
    """
    Supported:
      - float(low, high, log=true|false, step=<float|None>)
      - int(low, high, log=true|false, step=<int|None>)
      - categorical([a, b, c])
    """
    call = _parse_call(spec)
    fn = str(call.func.id)
    args = [_const(a) for a in call.args]
    kwargs = _call_kwargs(call)

    if fn == "float":
        if len(args) != 2:
            raise ValueError(f"float() expects (low, high), got {spec!r}")
        low, high = float(args[0]), float(args[1])
        log = bool(kwargs.get("log", False))
        step = kwargs.get("step", None)
        if step is None:
            return trial.suggest_float(name, low, high, log=log)
        return trial.suggest_float(name, low, high, step=float(step), log=log)

    if fn == "int":
        if len(args) != 2:
            raise ValueError(f"int() expects (low, high), got {spec!r}")
        low, high = int(args[0]), int(args[1])
        log = bool(kwargs.get("log", False))
        step = kwargs.get("step", 1)
        return trial.suggest_int(name, low, high, step=int(step), log=log)

    if fn == "categorical":
        if len(args) != 1 or not isinstance(args[0], list):
            raise ValueError(f"categorical() expects ([...]), got {spec!r}")
        return trial.suggest_categorical(name, args[0])

    raise ValueError(f"Unknown optuna spec function {fn!r} in {spec!r}")


def apply_optuna_suggestions(cfg: dict, trial) -> dict:
    cfg2 = deepcopy(cfg)

    def rec(obj: Any, prefix: str):
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                rec(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(list(obj)):
                rec(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            s = obj.strip()
            if "(" in s and s.endswith(")"):
                obj_parent, obj_key = _resolve_parent(cfg2, prefix)
                obj_parent[obj_key] = suggest_from_string(trial, prefix, s)

    rec(cfg2, "")
    return cfg2


def _resolve_parent(cfg: dict, path: str):
    if not path:
        raise ValueError("Empty path")
    parts = path.split(".")
    parent = cfg
    for p in parts[:-1]:
        if p.endswith("]"):
            raise ValueError("List paths are not supported for optuna suggestions")
        parent = parent[p]
    key = parts[-1]
    if key.endswith("]"):
        raise ValueError("List paths are not supported for optuna suggestions")
    return parent, key


__all__ = ["apply_optuna_suggestions", "suggest_from_string"]

