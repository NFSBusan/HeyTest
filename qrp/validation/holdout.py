"""The final holdout, behind a lock.

Every look is written to the registry BEFORE any holdout number is computed. By default a
holdout window can be opened once. A second look needs `force=True` and a reason, and every
report then marks that holdout as contaminated: once you have adjusted anything after seeing
holdout results, it is no longer out of sample.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..data.panel import Panel, Split
from ..registry import Registry


class HoldoutLocked(RuntimeError):
    pass


@dataclass
class HoldoutView:
    access_id: str
    panel: Panel          # full history (dev rows are needed to warm up indicators)
    start: int            # first scored holdout row
    times_opened: int


class HoldoutVault:
    def __init__(self, registry: Registry, full_panel: Panel, split: Split):
        self.reg = registry
        self._panel = full_panel
        self.split = split
        self.fp = full_panel.fingerprint(split.holdout_start, split.T)

    def times_opened(self) -> int:
        return len(self.reg.holdout_accesses(self.fp))

    def open(self, study_id: str, spec_hashes: list[str], force: bool = False, reason: str = "") -> HoldoutView:
        if not spec_hashes:
            raise HoldoutLocked("no candidates: only specs that passed every dev gate may see the holdout")
        prior = self.times_opened()
        if prior and not force:
            raise HoldoutLocked(
                f"this holdout window was already opened {prior} time(s). Looking again turns it into "
                "another dev set. Wait for new data (a fresh holdout), or pass force=True with a reason; "
                "the report will then label holdout results as contaminated.")
        if force and not reason:
            raise HoldoutLocked("a forced holdout look needs a written reason")
        aid = self.reg.log_holdout_access(self.fp, study_id, list(spec_hashes), force, reason)
        return HoldoutView(aid, self._panel, self.split.holdout_start, prior + 1)
