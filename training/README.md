# Training Pipeline — MapGen, Dense Rewards & PLR Curriculum

This directory contains the procedural map generation, dense reward shaping, and self-evolving curriculum modules for Track 1 ("Can It Play DOOM?").

## 1. Procedural 4-Tier Map Generator (`training/mapgen/`)
- **`udmf_builder.py`**: Watertight ZDoom UDMF (`TEXTMAP` + `ENDMAP`) PWAD writer using non-overlapping orthogonal sectors. Automatically splits shared portals into 2-sided linedefs (doors, locked doors, height steps, exit alcoves) and 1-sided walls/pillars.
- **`tiers.py`**: `MapGen.sample(tier, knobs, seed, out_dir, scaffolding_alpha)` generates randomized, non-hardcoded maps across 4 tiers and runs a generation-time BFS solvability check over the room-key-lock graph:
  - **Tier 1:** 1–2 open arena rooms with randomized pillars, enemies, pickups, spawn, and exit alcove.
  - **Tier 2:** 3–5 connected rooms with regular doors (`special=12`, ray `hit_type == 2`).
  - **Tier 3:** Keycard backtracking (Branch A: keycard; Branch B: locked door `special=13`, `arg3 ∈ {1,2,3}`, ray `hit_type == 3` → exit).
  - **Tier 4:** 5–8 room gauntlet with 1–2 sequential keycard locks, ambush enemies, and tighter ammo.

## 2. Dense Privileged Reward Shaper (`training/rewards.py`)
`RewardShaper` uses `env.privileged_state()` and `env.sector_polygons()` dynamically on any map (zero hardcoded coordinates or sector IDs):
- Exit: `+15.0 + 3.0 * (rem_ticks / episode_ticks)`
- New sector visited: `+0.40 * clamp(area / 512^2, 0.2, 1.5)`
- Keycard pickup: `+4.00`
- Door opened: `+0.60` | Locked door unlocked: `+5.00`
- Enemy kill: `+1.00` | Item pickup: `+0.15`
- Penalties: damage (`-0.02/HP`), death (`-4.00`), door-reversal bounce (`-0.30`), wall-scrape (`-0.02/step`), step cost (`-0.002`)

## 3. Self-Evolving Curriculum (`training/curriculum_plr.py`)
`PLRBuffer` maintains `256` active procedural seeds (`seed ∈ [100_000, 999_999]`, strictly disjoint from held-out `eval_suite.json` seeds `∈ [1_000, 9_999]`), prioritized by `mean |GAE|` + staleness bonus, paired with `scaffolding_alpha` (`1.0 → 0.0` over `12M` steps).

## 4. Commands
```bash
# Run unit & contract tests
uv run pytest tests/test_mapgen_and_rewards.py

# Stress-test 500 generated maps across Tiers 1-4 in DoomEnv
uv run python -m training.mapgen.verify_maps --count 500 --steps 10
```
