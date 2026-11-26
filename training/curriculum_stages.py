"""Staged (easy -> hard) curriculum for the DOOM PPO trainer.

Replaces the hard-biased PLR sampling (tier_weights [1, 1.5, 2.5, 3.0] -> ~75% Tier 3/4 maps)
with an explicit ladder the agent has to climb:

    0  single_map   ONE fixed map, no enemies (learn: find the exit)
    1  easy         Tier 1, random seeds, a few weak enemies
    2  medium       Tier 2, random seeds, doors
    3  hard         Tier 3 (keycards) + Tier 4 (multi-lock) mix
    4  random       every tier mixed (final generalisation stage)

A stage is "mastered" when the rolling exit-rate over the last `window` episodes reaches its
threshold (after `min_episodes`), or when `max_steps_per_stage` is spent (so it never stalls).
`stage_lock` pins a run to one stage: used so teammates can train one specialist each in parallel.

Training seeds always come from [100_000, 999_999] (disjoint from held-out [1_000, 9_999]).
Uses no network / no LLM. It is a plain feedback rule on the training GPU run's own telemetry.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

TRAIN_SEED_MIN = 100_000
TRAIN_SEED_MAX = 999_999
FIXED_MAP_SEED = 123_457  # the single map of stage 0 (always the same layout)


@dataclass
class StageSpec:
    name: str
    tiers: Sequence[int]
    tier_probs: Sequence[float]
    knobs: dict[str, Any]
    exit_threshold: float
    scaffolding_alpha: float
    fixed_seed: Optional[int] = None


DEFAULT_STAGES: list[StageSpec] = [
    StageSpec(
        name="single_map",
        tiers=[1],
        tier_probs=[1.0],
        knobs={"scenario": "open_arena", "num_rooms": 1, "enemies_per_room": 0.0, "ammo_mult": 1.5},
        exit_threshold=0.80,
        scaffolding_alpha=1.0,
        fixed_seed=FIXED_MAP_SEED,
    ),
    StageSpec(
        name="easy",
        tiers=[1],
        tier_probs=[1.0],
        knobs={"num_rooms": 2, "enemies_per_room": 1.0, "ammo_mult": 1.3},
        exit_threshold=0.60,
        scaffolding_alpha=0.8,
    ),
    StageSpec(
        name="medium",
        tiers=[1, 2],
        tier_probs=[0.25, 0.75],
        knobs={"enemies_per_room": 1.5, "ammo_mult": 1.1},
        exit_threshold=0.45,
        scaffolding_alpha=0.5,
    ),
    StageSpec(
        name="hard",
        tiers=[2, 3, 4],
        tier_probs=[0.2, 0.5, 0.3],
        knobs={},
        exit_threshold=0.30,
        scaffolding_alpha=0.25,
    ),
    StageSpec(
        name="random",
        tiers=[1, 2, 3, 4],
        tier_probs=[0.25, 0.25, 0.25, 0.25],
        knobs={},
        exit_threshold=1.01,  # final stage never "completes"
        scaffolding_alpha=0.0,
    ),
]


@dataclass
class StageCurriculum:
    stages: list[StageSpec] = field(default_factory=lambda: list(DEFAULT_STAGES))
    stage_idx: int = 0
    stage_lock: Optional[int] = None
    window: int = 64
    min_episodes: int = 64
    min_steps_per_stage: int = 100_000
    max_steps_per_stage: int = 1_500_000
    seed: int = 7
    state_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._exits: deque[float] = deque(maxlen=self.window)
        self._episodes_in_stage = 0
        self._stage_start_step = 0
        if self.stage_lock is not None:
            self.stage_idx = int(self.stage_lock)
        self.stage_idx = max(0, min(len(self.stages) - 1, int(self.stage_idx)))
        self._used: set[int] = set()

    # ------------------------------------------------------------------ queries
    @property
    def spec(self) -> StageSpec:
        return self.stages[self.stage_idx]

    @property
    def name(self) -> str:
        return f"{self.stage_idx}:{self.spec.name}"

    def exit_rate(self) -> float:
        return float(np.mean(self._exits)) if self._exits else 0.0

    # ------------------------------------------------------------------ sampling
    def _fresh_seed(self) -> int:
        for _ in range(64):
            s = int(self.rng.integers(TRAIN_SEED_MIN, TRAIN_SEED_MAX + 1))
            if s not in self._used:
                self._used.add(s)
                return s
        return int(self.rng.integers(TRAIN_SEED_MIN, TRAIN_SEED_MAX + 1))

    def sample(self) -> dict[str, Any]:
        """Return a level dict compatible with `PLRBuffer.sample_level` consumers in ppo.py."""
        sp = self.spec
        p = np.asarray(sp.tier_probs, dtype=np.float64)
        tier = int(self.rng.choice(list(sp.tiers), p=p / p.sum()))
        seed = sp.fixed_seed if sp.fixed_seed is not None else self._fresh_seed()
        return {
            "level_id": f"stage{self.stage_idx}_t{tier}_s{seed}",
            "tier": tier,
            "knobs": dict(sp.knobs),
            "seed": int(seed),
            "scaffolding_alpha": float(sp.scaffolding_alpha),
        }

    # ------------------------------------------------------------------ feedback
    def report_episode(self, exit_reached: bool) -> None:
        self._exits.append(1.0 if exit_reached else 0.0)
        self._episodes_in_stage += 1

    def maybe_advance(self, global_step: int) -> Optional[str]:
        """Advance to the next stage when mastered (or out of budget). Returns a log line if it moved."""
        if self.stage_lock is not None or self.stage_idx >= len(self.stages) - 1:
            return None
        if self._stage_start_step == 0 and self._episodes_in_stage == 0:
            self._stage_start_step = global_step
        spent = global_step - self._stage_start_step
        mastered = (
            self._episodes_in_stage >= self.min_episodes
            and spent >= self.min_steps_per_stage
            and self.exit_rate() >= self.spec.exit_threshold
        )
        out_of_budget = spent >= self.max_steps_per_stage
        if not (mastered or out_of_budget):
            return None
        old = self.name
        old_rate = self.exit_rate()
        self.stage_idx += 1
        self._exits.clear()
        self._episodes_in_stage = 0
        self._stage_start_step = global_step
        why = "MASTERED" if mastered else "step budget spent"
        msg = f"[Stage] {old} -> {self.name} ({why}; exit={old_rate * 100:.0f}% @ step {global_step:,})"
        self.save()
        return msg

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.write_text(
                json.dumps({"stage_idx": self.stage_idx, "stage": self.spec.name}), encoding="utf-8"
            )
        except OSError:
            pass
