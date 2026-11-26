"""Prioritized Level Replay (PLR) curriculum & scaffolding annealing schedule.

Self-evolving map distribution: tracks 256 active procedural seeds across Tiers 1-4,
prioritizing maps with high learning potential (mean |GAE| + staleness bonus),
while keeping training seeds (>= 100_000) strictly disjoint from held-out seeds ([1_000, 9_999]).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

TRAIN_SEED_MIN = 100_000
TRAIN_SEED_MAX = 999_999


@dataclass
class PLREntry:
    level_id: str
    tier: int
    knobs: dict[str, Any]
    seed: int
    score_gae: float = 1.0
    last_seen_update: int = 0
    visits: int = 0


class PLRBuffer:
    """Prioritized Level Replay buffer + scaffolding annealing."""

    def __init__(
        self,
        capacity: int = 256,
        replay_prob: float = 0.70,
        staleness_coef: float = 0.10,
        temperature: float = 1.0,
        scaffolding_total_steps: int = 12_000_000,
        seed: int = 42,
    ) -> None:
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.staleness_coef = staleness_coef
        self.temperature = temperature
        self.scaffolding_total_steps = max(1, scaffolding_total_steps)
        self.rng = random.Random(seed)
        self.entries: dict[int, PLREntry] = {}
        self.update_idx: int = 0

    def scaffolding_alpha(self, global_step: int) -> float:
        """Linearly anneal scaffolding from 1.0 -> 0.0 over scaffolding_total_steps."""
        frac = max(0.0, min(1.0, global_step / float(self.scaffolding_total_steps)))
        return 1.0 - frac

    def _tier_weights(self, global_step: int) -> list[float]:
        """Shift tier sampling smoothly from Tier 1-2 early to Tier 3-4 later."""
        progress = max(0.0, min(1.0, global_step / float(self.scaffolding_total_steps)))
        w1 = max(0.10, 0.45 * (1.0 - progress))
        w2 = 0.30
        w3 = 0.15 + 0.25 * progress
        w4 = 0.10 + 0.20 * progress
        return [w1, w2, w3, w4]

    def _default_knobs(self, tier: int) -> dict[str, Any]:
        if tier == 1:
            return {"num_rooms": self.rng.choice([1, 2]), "enemies_per_room": self.rng.randint(2, 4), "ammo_mult": 1.2}
        if tier == 2:
            return {"num_rooms": self.rng.randint(3, 5), "enemies_per_room": self.rng.randint(1, 3), "ammo_mult": 1.1}
        if tier == 3:
            return {"num_rooms": self.rng.randint(4, 6), "num_keys": 1, "enemies_per_room": self.rng.randint(1, 3), "ammo_mult": 1.0}
        return {"num_rooms": self.rng.randint(5, 8), "num_keys": self.rng.choice([1, 2]), "enemies_per_room": self.rng.randint(2, 4), "ammo_mult": 0.9}

    def sample_level(self, global_step: int = 0) -> tuple[PLREntry, float]:
        """Sample either a prioritized replay level or a fresh procedural level."""
        alpha = self.scaffolding_alpha(global_step)
        if len(self.entries) >= 16 and self.rng.random() < self.replay_prob:
            entries = list(self.entries.values())
            weights = []
            for e in entries:
                staleness = max(0, self.update_idx - e.last_seen_update)
                score = max(1e-4, e.score_gae) + self.staleness_coef * (staleness / max(1, self.capacity))
                weights.append(score ** (1.0 / max(1e-3, self.temperature)))
            chosen = self.rng.choices(entries, weights=weights, k=1)[0]
            chosen.last_seen_update = self.update_idx
            chosen.visits += 1
            return chosen, alpha

        tier = self.rng.choices([1, 2, 3, 4], weights=self._tier_weights(global_step), k=1)[0]
        seed = self.rng.randint(TRAIN_SEED_MIN, TRAIN_SEED_MAX)
        while seed in self.entries:
            seed = self.rng.randint(TRAIN_SEED_MIN, TRAIN_SEED_MAX)
        entry = PLREntry(
            level_id=f"proc_t{tier}_s{seed}",
            tier=tier,
            knobs=self._default_knobs(tier),
            seed=seed,
            score_gae=1.0,
            last_seen_update=self.update_idx,
            visits=1,
        )
        self._insert(entry)
        return entry, alpha

    def _insert(self, entry: PLREntry) -> None:
        if len(self.entries) >= self.capacity:
            # Evict lowest-priority entry
            victim_seed = min(
                self.entries,
                key=lambda s: self.entries[s].score_gae + self.staleness_coef * ((self.update_idx - self.entries[s].last_seen_update) / max(1, self.capacity)),
            )
            del self.entries[victim_seed]
        self.entries[entry.seed] = entry

    def update_score(self, seed: int, mean_abs_gae: float) -> None:
        """Update learning-potential priority for a seed after a rollout."""
        self.update_idx += 1
        if seed in self.entries:
            e = self.entries[seed]
            # Exponential moving average of |GAE|
            e.score_gae = 0.5 * e.score_gae + 0.5 * float(max(0.0, mean_abs_gae))
            e.last_seen_update = self.update_idx
