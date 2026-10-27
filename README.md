# HeyTest

Two research projects:

- **`qrp/`: quant research platform** (Strategy 2). It generates, backtests and validates strategy
  families on daily bars, counts every configuration it tries, keeps a locked final holdout, can run a
  Claude-assisted research loop with your approval, and paper-trades strategies that pass. Start with
  [`docs/PLATFORM.md`](docs/PLATFORM.md) (plan, providers, costs, setup), and see
  [`docs/IMPROVED_PROMPT.md`](docs/IMPROVED_PROMPT.md) for the revised strategy prompt.
- **`pmct/`: Polymarket counter-trading research**. See [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

Quick start (offline, simulated data):

```bash
pip install -e ".[dev,claude]"
pytest -q
python -m qrp research --budget 300
```

Then open `reports/dashboard_<study>.html`. Results on simulated data test the software only.
