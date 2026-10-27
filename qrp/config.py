"""Config loading. Secrets come from environment variables (or a local .env), never from YAML."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "qrp.yaml"


class ConfigError(ValueError):
    pass


class Config(dict):
    """A dict with attribute-style section access: cfg.costs["half_spread_bps"]."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with dotted-key overrides applied, e.g. {"data.provider": "csv"}."""
        out = copy.deepcopy(dict(self))
        for key, value in overrides.items():
            node = out
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return Config(out)


REQUIRED = {
    "project": ["name", "seed"],
    "storage": ["root", "db_path", "paper_db_path"],
    "data": ["provider", "timeframe", "start", "end", "benchmark", "symbols"],
    "splits": ["holdout_fraction", "embargo_days", "warmup_days", "wf_folds", "wf_min_train_years"],
    "execution": ["fill", "allow_short", "max_gross_leverage", "max_weight"],
    "costs": ["capital", "commission_bps", "half_spread_bps", "impact_coef", "borrow_bps_annual",
              "stress_multipliers"],
    "search": ["max_trials_per_family", "families", "candidates_per_family", "dedup_corr"],
    "gates": ["min_trades", "min_dsr", "max_pbo"],
    "validation": ["bootstrap_reps", "block_len", "cscv_splits", "mc_reps"],
}


def validate(cfg: dict) -> None:
    for section, keys in REQUIRED.items():
        if section not in cfg:
            raise ConfigError(f"config is missing section '{section}'")
        for k in keys:
            if k not in cfg[section]:
                raise ConfigError(f"config is missing '{section}.{k}'")
    s = cfg["splits"]
    if not 0.05 <= s["holdout_fraction"] <= 0.5:
        raise ConfigError("splits.holdout_fraction must be between 0.05 and 0.5")
    if cfg["execution"]["fill"] not in ("next_open", "next_close"):
        raise ConfigError("execution.fill must be next_open or next_close")
    if cfg["validation"]["cscv_splits"] % 2:
        raise ConfigError("validation.cscv_splits must be even")
    if cfg["data"]["benchmark"] not in cfg["data"]["symbols"]:
        raise ConfigError("data.benchmark must be one of data.symbols")
    if cfg.get("paper", {}).get("live_trading_enabled"):
        raise ConfigError("live trading is not implemented; paper.live_trading_enabled must be false")


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    load_dotenv()
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    cfg = Config(yaml.safe_load(p.read_text()))
    if overrides:
        cfg = cfg.with_overrides(overrides)
    validate(cfg)
    return cfg


def secret(name: str, required: bool = True) -> str | None:
    """Read a secret from the environment. Never log the value."""
    v = os.environ.get(name)
    if required and not v:
        raise ConfigError(f"environment variable {name} is not set (see .env.example)")
    return v
