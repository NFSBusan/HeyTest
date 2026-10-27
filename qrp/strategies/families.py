"""Strategy families: which blocks may combine, and the generator that enumerates them.

A family groups strategies that share an economic idea. Multiple-testing corrections, PBO
and the SPA test are computed per family, and trial budgets are capped per family, so a
family is the unit of "one research question".
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from .blocks import BLOCKS
from .spec import canonical, spec_hash, validate_spec


@dataclass(frozen=True)
class Family:
    name: str
    hypothesis: str
    signals: tuple[str, ...]
    filters: tuple[str, ...]
    exits: tuple[str, ...]
    sizings: tuple[str, ...]


FAMILIES: dict[str, Family] = {f.name: f for f in [
    Family("trend",
           "Assets that have been rising keep rising for a while (time-series momentum / trend "
           "following, documented across asset classes over a century). Edge is weak and comes in "
           "bursts; it should protect in long bear markets.",
           ("ma_cross", "tsmom", "donchian", "macd"),
           ("none", "vol_regime", "adx_above", "benchmark_regime"),
           ("hold", "atr_trail"),
           ("slot", "inverse_vol", "vol_target")),
    Family("mean_reversion",
           "Short-term oversold moves in liquid funds partly reverse within days (liquidity "
           "provision). Most of the documented edge is in equity indices, inside uptrends.",
           ("rsi_reversion", "bollinger_reversion", "stoch_reversion"),
           ("none", "trend_regime", "adx_below", "benchmark_regime"),
           ("hold", "time_stop", "atr_trail"),
           ("slot", "equal_active", "inverse_vol")),
    Family("cross_sectional",
           "Relative strength: funds that outperformed peers over 3-12 months keep outperforming "
           "for a few months; very short-term losers bounce (short-term reversal).",
           ("xs_momentum", "xs_reversal"),
           ("none", "benchmark_regime", "vol_regime"),
           ("hold",),
           ("equal_active", "inverse_vol")),
]}


def block_variants(name: str) -> list[dict]:
    b = BLOCKS[name]
    if not b.grid:
        return [{"name": name}]
    keys = list(b.grid)
    out = []
    for combo in itertools.product(*(b.grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        if b.constraint is None or b.constraint(params):
            out.append({"name": name, **params})
    return out


def enumerate_family(family: str, direction: str = "long") -> list[dict]:
    f = FAMILIES[family]
    slots = [
        [v for n in f.signals for v in block_variants(n)],
        [v for n in f.filters for v in block_variants(n)],
        [v for n in f.exits for v in block_variants(n)],
        [v for n in f.sizings for v in block_variants(n)],
    ]
    return [canonical({"family": family, "direction": direction, "signal": s, "filter": fl,
                       "exit": e, "sizing": z})
            for s, fl, e, z in itertools.product(*slots)]


def sample_family(family: str, budget: int, seed: int, exclude: set[str] = frozenset(),
                  direction: str = "long") -> list[dict]:
    """Up to `budget` distinct, untested configs, sampled uniformly (reproducible by seed)."""
    specs = enumerate_family(family, direction)
    rng = random.Random(f"{seed}:{family}:{direction}")
    rng.shuffle(specs)
    out = []
    for s in specs:
        if spec_hash(s) in exclude:
            continue
        out.append(s)
        if len(out) >= budget:
            break
    return out


def neighbours(spec: dict) -> list[dict]:
    """Specs that differ from `spec` by one grid step in one numeric parameter."""
    spec = validate_spec(spec)
    out = []
    for slot in ("signal", "filter", "exit", "sizing"):
        b = BLOCKS[spec[slot]["name"]]
        for k, grid in b.grid.items():
            v = spec[slot][k]
            if isinstance(v, str) or len(grid) < 2:
                continue
            g = sorted(grid)
            # Values off the grid (e.g. from Optuna or Claude) step to the nearest grid points.
            lower = [x for x in g if x < v]
            higher = [x for x in g if x > v]
            for nv in ([lower[-1]] if lower else []) + ([higher[0]] if higher else []):
                cand = {**spec, slot: {**spec[slot], k: nv}}
                params = {kk: vv for kk, vv in cand[slot].items() if kk != "name"}
                if b.constraint is None or b.constraint(params):
                    out.append(canonical(cand))
    return out


def family_size(family: str) -> int:
    return len(enumerate_family(family))
