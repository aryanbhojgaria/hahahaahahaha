"""Dense training-only reward shaper using DoomEnv privileged_state() and sector_polygons().

Works dynamically on any procedural or unseen WAD without hardcoded coordinates or sector IDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    exit_base: float = 15.0
    exit_speed_bonus: float = 3.0
    new_sector: float = 0.40
    key_pickup: float = 4.00
    door_open: float = 0.60
    locked_door_unlock: float = 5.00
    kill: float = 1.00
    item_pickup: float = 0.15
    damage_per_hp: float = -0.02
    death: float = -4.00
    door_reversal_bounce: float = -0.30
    stuck_wall_scrape: float = -0.02
    step_cost: float = -0.002


@dataclass
class _SectorPoly:
    sector_id: int
    segments: np.ndarray  # (K, 4) [x1, y1, x2, y2]
    bbox: tuple[float, float, float, float]
    area_weight: float


def _point_in_segments(x: float, y: float, segs: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test over unordered boundary segments."""
    inside = False
    for x1, y1, x2, y2 in segs:
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


class RewardShaper:
    """Computes dense per-step rewards during training using privileged_state()."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.w = weights or RewardWeights()
        self._sectors: list[_SectorPoly] = []
        self._visited_sectors: set[int] = set()
        self._opened_doors: set[int] = set()
        self._prev_state: dict | None = None
        self._prev_ceil_vel: np.ndarray | None = None
        self._stuck_steps: int = 0

    def reset(self, env) -> None:
        """Initialize sector geometry and episode trackers from env after env.reset()."""
        raw_polys = env.sector_polygons()
        self._sectors = []
        for sec_id, segs in raw_polys:
            if len(segs) == 0:
                continue
            xs = np.concatenate([segs[:, 0], segs[:, 2]])
            ys = np.concatenate([segs[:, 1], segs[:, 3]])
            min_x, max_x = float(xs.min()), float(xs.max())
            min_y, max_y = float(ys.min()), float(ys.max())
            approx_area = max(1.0, (max_x - min_x) * (max_y - min_y))
            area_weight = max(0.2, min(1.5, approx_area / (512.0 * 512.0)))
            self._sectors.append(_SectorPoly(sec_id, segs, (min_x, min_y, max_x, max_y), area_weight))

        self._visited_sectors.clear()
        self._opened_doors.clear()
        self._stuck_steps = 0
        state = env.privileged_state()
        self._prev_state = state
        ceil = state.get("sector_ceiling")
        self._prev_ceil_vel = np.zeros_like(ceil) if ceil is not None else None
        self._mark_sector(state["x"], state["y"])

    def _mark_sector(self, x: float, y: float) -> float:
        """Return new-sector reward if (x, y) enters a previously unvisited sector."""
        for sp in self._sectors:
            if sp.sector_id in self._visited_sectors:
                continue
            min_x, min_y, max_x, max_y = sp.bbox
            if min_x <= x <= max_x and min_y <= y <= max_y:
                if _point_in_segments(x, y, sp.segments):
                    self._visited_sectors.add(sp.sector_id)
                    return self.w.new_sector * sp.area_weight
        return 0.0

    def step(self, env, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute dense shaped reward and breakdown dict for one env step."""
        if self._prev_state is None:
            raise RuntimeError("Call RewardShaper.reset(env) before step().")

        cur = env.privileged_state()
        prev = self._prev_state
        r_total = self.w.step_cost
        info: dict[str, float] = {}

        # 1. Exit & Death
        if cur["exit_reached"]:
            rem = max(0, cur["episode_ticks"] - cur["tic"])
            frac = rem / max(1, cur["episode_ticks"])
            r_exit = self.w.exit_base + self.w.exit_speed_bonus * frac
            r_total += r_exit
            info["r_exit"] = r_exit

        if cur["dead"]:
            r_total += self.w.death
            info["r_death"] = self.w.death

        # 2. Sector Exploration
        r_sec = self._mark_sector(cur["x"], cur["y"])
        if r_sec != 0.0:
            r_total += r_sec
            info["r_sector"] = r_sec

        # 3. Keycard Pickups
        keys_delta = int(np.sum(cur["keys"]) - np.sum(prev["keys"]))
        if keys_delta > 0:
            r_key = self.w.key_pickup * keys_delta
            r_total += r_key
            info["r_key"] = r_key

        # 4. Door Open / Locked Door Unlock / Door Reversal Bounce
        ceil_now = cur.get("sector_ceiling")
        ceil_prev = prev.get("sector_ceiling")
        floor_now = cur.get("sector_floor")
        door_lock = cur.get("door_lock")
        if (
            ceil_now is not None
            and ceil_prev is not None
            and floor_now is not None
            and door_lock is not None
            and len(ceil_now) == len(ceil_prev)
        ):
            ceil_vel = ceil_now - ceil_prev
            use_pressed = int(action[7]) == 1
            for s_idx in range(len(ceil_now)):
                is_door_sector = int(door_lock[s_idx]) >= 0 and (
                    ceil_vel[s_idx] != 0 or s_idx in self._opened_doors
                )
                # First time a door sector rises above closed threshold
                if (
                    s_idx not in self._opened_doors
                    and ceil_vel[s_idx] > 1.0
                    and ceil_now[s_idx] > floor_now[s_idx] + 8.0
                ):
                    self._opened_doors.add(s_idx)
                    if int(door_lock[s_idx]) > 0:
                        r_total += self.w.locked_door_unlock
                        info["r_unlock"] = info.get("r_unlock", 0.0) + self.w.locked_door_unlock
                    else:
                        r_total += self.w.door_open
                        info["r_door"] = info.get("r_door", 0.0) + self.w.door_open

                # Penalize spamming USE to reverse an already-opening door
                if (
                    use_pressed
                    and is_door_sector
                    and self._prev_ceil_vel is not None
                    and self._prev_ceil_vel[s_idx] > 0.5
                    and ceil_vel[s_idx] < -0.5
                ):
                    r_total += self.w.door_reversal_bounce
                    info["r_door_bounce"] = info.get("r_door_bounce", 0.0) + self.w.door_reversal_bounce

            self._prev_ceil_vel = ceil_vel.copy()

        # 5. Combat & Pickups & Damage
        kills_delta = max(0, int(cur["kills"]) - int(prev["kills"]))
        if kills_delta > 0:
            r_kill = self.w.kill * kills_delta
            r_total += r_kill
            info["r_kill"] = r_kill

        items_delta = max(0, int(cur["items"]) - int(prev["items"]))
        if items_delta > 0:
            r_item = self.w.item_pickup * items_delta
            r_total += r_item
            info["r_item"] = r_item

        dmg_delta = max(0.0, float(cur["damage_taken"]) - float(prev["damage_taken"]))
        if dmg_delta > 0.0:
            r_dmg = self.w.damage_per_hp * dmg_delta
            r_total += r_dmg
            info["r_damage"] = r_dmg

        # 6. Wall-scrape / stuck penalty
        dist_moved = math.hypot(cur["x"] - prev["x"], cur["y"] - prev["y"])
        if int(action[0]) == 1 and dist_moved < 2.0 and not cur["finished"]:
            self._stuck_steps += 1
            if self._stuck_steps > 8:
                r_total += self.w.stuck_wall_scrape
                info["r_stuck"] = self.w.stuck_wall_scrape
        else:
            self._stuck_steps = 0

        self._prev_state = cur
        return float(r_total), info

    @property
    def visited_sector_count(self) -> int:
        return len(self._visited_sectors)
