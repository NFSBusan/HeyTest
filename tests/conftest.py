import pytest

from qrp.config import load_config

SYMS = ["SPY", "QQQ", "TLT", "GLD", "EFA", "IWM", "XLE", "XLK"]


def make_cfg(tmp_path, **over):
    base = {
        "storage.root": str(tmp_path / "store"),
        "storage.db_path": str(tmp_path / "r.duckdb"),
        "storage.paper_db_path": str(tmp_path / "p.duckdb"),
        "storage.reports_dir": str(tmp_path / "reports"),
        "data.symbols": SYMS,
        "data.start": "2008-01-01",
        "data.end": "2023-12-31",
        "search.max_trials_per_family": 40,
        "search.n_jobs": 1,
        "search.candidates_per_family": 2,
        "validation.bootstrap_reps": 100,
        "validation.mc_reps": 100,
        "splits.wf_folds": 3,
        "splits.wf_min_train_years": 3,
    }
    base.update(over)
    return load_config(overrides=base)


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)
