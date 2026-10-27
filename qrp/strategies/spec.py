"""Strategy specifications: plain JSON-able dicts with a stable hash.

Example:
    {"family": "trend", "direction": "long",
     "signal": {"name": "ma_cross", "fast": 20, "slow": 100, "kind": "sma"},
     "filter": {"name": "none"},
     "exit":   {"name": "atr_trail", "n": 14, "mult": 3},
     "sizing": {"name": "inverse_vol", "n": 60}}

The hash identifies a configuration across studies, so the registry can skip re-testing it
and can count every distinct configuration ever tried on the same data.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..indicators import IndicatorCache
from .blocks import BLOCKS

SLOTS = ("signal", "filter", "exit", "sizing")
SLOT_KIND = {"signal": "signal", "filter": "filter", "exit": "exit", "sizing": "sizing"}


class SpecError(ValueError):
    pass


def _norm(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 10)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    raise SpecError(f"unsupported value {v!r}")


def canonical(spec: dict) -> dict:
    out = {"family": spec["family"], "direction": spec.get("direction", "long")}
    for slot in SLOTS:
        out[slot] = _norm(dict(spec.get(slot) or {"name": {"filter": "none", "exit": "hold",
                                                            "sizing": "slot"}.get(slot, "")}))
    return out


def spec_hash(spec: dict) -> str:
    return hashlib.sha1(json.dumps(canonical(spec), sort_keys=True).encode()).hexdigest()[:16]


def describe(spec: dict) -> str:
    s = canonical(spec)

    def part(d):
        args = ",".join(f"{k}={v}" for k, v in d.items() if k != "name")
        return f"{d['name']}({args})" if args else d["name"]
    return " | ".join(part(s[k]) for k in SLOTS) + f" [{s['direction']}]"


def validate_spec(spec: dict, check_bounds: bool = False) -> dict:
    s = canonical(spec)
    if s["direction"] not in ("long", "long_short"):
        raise SpecError("direction must be long or long_short")
    for slot in SLOTS:
        name = s[slot].get("name")
        b = BLOCKS.get(name)
        if b is None or b.kind != SLOT_KIND[slot]:
            raise SpecError(f"{slot}: unknown block {name!r}")
        params = {k: v for k, v in s[slot].items() if k != "name"}
        expected = set(b.grid)
        if set(params) != expected:
            raise SpecError(f"{slot}.{name}: expected params {sorted(expected)}, got {sorted(params)}")
        if b.constraint and not b.constraint(params):
            raise SpecError(f"{slot}.{name}: parameters violate constraint {params}")
        if check_bounds:
            for k, v in params.items():
                if k in b.bounds and not isinstance(v, str):
                    lo, hi = b.bounds[k]
                    if not lo <= v <= hi:
                        raise SpecError(f"{slot}.{name}.{k}={v} outside allowed range [{lo}, {hi}]")
                elif isinstance(v, str) and v not in b.grid[k]:
                    raise SpecError(f"{slot}.{name}.{k}={v!r} not one of {b.grid[k]}")
    return s


def target_weights(spec: dict, cache: IndicatorCache, max_weight: float, max_gross: float,
                   allow_short: bool) -> np.ndarray:
    """Spec -> (T, N) target weights decided at each day's close (executed next bar by the engine)."""
    s = canonical(spec)
    short = s["direction"] == "long_short"
    if short and not allow_short:
        raise SpecError("spec is long_short but execution.allow_short is false")
    p = cache.p

    def args(slot):
        return {k: v for k, v in s[slot].items() if k != "name"}

    D = BLOCKS[s["signal"]["name"]].fn(cache, short, **args("signal"))
    D = np.where(p.tradable, D, 0.0)
    ok_long, ok_short = BLOCKS[s["filter"]["name"]].fn(cache, **args("filter"))
    D = np.where(D > 0, D * ok_long, np.where(D < 0, D * ok_short, 0.0))
    D = BLOCKS[s["exit"]["name"]].fn(cache, np.ascontiguousarray(D, dtype=np.float64), **args("exit"))
    D = np.where(p.tradable, D, 0.0)
    w = BLOCKS[s["sizing"]["name"]].fn(cache, D, **args("sizing"))
    w = np.nan_to_num(w)
    if not short:
        w = np.maximum(w, 0.0)
    w = np.clip(w, -max_weight, max_weight)
    gross = np.abs(w).sum(axis=1, keepdims=True)
    scale = np.where(gross > max_gross, max_gross / np.where(gross > 0, gross, 1.0), 1.0)
    return w * scale
