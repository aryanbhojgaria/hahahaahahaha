# Training Pipeline — Architecture, MapGen, Dense Rewards & PLR Curriculum

This directory contains the neural architecture, preprocessor, odometry grid, procedural map generator, dense reward shaping, and self-evolving curriculum modules for Track 1 ("Can It Play DOOM?").

## 1. Neural Architecture & Spatial Memory (`training/`)
- **`preprocessing.py`**: Observation scaler and legal action mask. Packs 64 rays into (8, 64), entities into (32, 5), vitals into 36-dim vector, and 64x48 RGB+depth buffer. Implements door-reversal `USE` cooldown masking and ammo `ATTACK` masking.
- **`odometry_grid.py`**: 360° ray circular cross-correlation scan matching to estimate yaw rotation $d_\theta$, integrated with forward/right velocity to affine-warp a 32x32 6-channel semantic occupancy grid (`visited`, `wall`, `door`, `locked_door`, `item`, `enemy`).
- **`model.py` (`HybridDoomNet`)**:
  - Circular 1D CNN over 64 rays (`padding_mode="circular"`).
  - Masked Multi-Head Attention over <= 32 entities (NaN-safe for 0 entities).
  - Vitals MLP (36 -> 48).
  - Compact RGB+depth CNN with modality dropout ($p=0.25$).
  - Egocentric Grid 2D CNN (32x32x6 -> 64).
  - Recurrent core: `GRU(256, 256)`.
  - 4 Objective Memory Slots rotated by odometry $-d_\theta$ with gated GRU writes.
  - 9 Bernoulli action heads with observation action masking, critic value head, and auxiliary heads.

## 2. Procedural 4-Tier Map Generator (`training/mapgen/`)
- **`udmf_builder.py`**: Watertight ZDoom UDMF (`TEXTMAP` + `ENDMAP`) PWAD writer using non-overlapping orthogonal sectors. Automatically splits shared portals into 2-sided linedefs (doors, locked doors, height steps, exit alcoves) and 1-sided walls/pillars.
- **`tiers.py`**: `MapGen.sample(tier, knobs, seed, out_dir, scaffolding_alpha)` generates randomized, non-hardcoded maps across 4 tiers and runs a generation-time BFS solvability check over the room-key-lock graph:
  - **Tier 1:** 1–2 open arena rooms with randomized pillars, enemies, pickups, spawn, and exit alcove.
  - **Tier 2:** 3–5 connected rooms with regular doors (`special=12`, ray `hit_type == 2`).
  - **Tier 3:** Keycard backtracking (Branch A: keycard; Branch B: locked door `special=13`, `arg3 ∈ {1,2,3}`, ray `hit_type == 3` → exit).
  - **Tier 4:** 5–8 room gauntlet with 1–2 sequential keycard locks, ambush enemies, and tighter ammo.

## 3. Dense Privileged Reward Shaper (`training/rewards.py`)
`RewardShaper` uses `env.privileged_state()` and `env.sector_polygons()` dynamically on any map (zero hardcoded coordinates or sector IDs):
- Exit: `+15.0 + 3.0 * (rem_ticks / episode_ticks)`
- New sector visited: `+0.40 * clamp(area / 512^2, 0.2, 1.5)`
- Keycard pickup: `+4.00`
- Door opened: `+0.60` | Locked door unlocked: `+5.00`
- Enemy kill: `+1.00` | Item pickup: `+0.15`
- Penalties: damage (`-0.02/HP`), death (`-4.00`), door-reversal bounce (`-0.30`), wall-scrape (`-0.02/step`), step cost (`-0.002`)

## 4. Self-Evolving Curriculum (`training/curriculum_plr.py`)
`PLRBuffer` maintains `256` active procedural seeds (`seed ∈ [100_000, 999_999]`, strictly disjoint from held-out `eval_suite.json` seeds `∈ [1_000, 9_999]`), prioritized by `mean |GAE|` + staleness bonus, paired with `scaffolding_alpha` (`1.0 → 0.0` over `12M` steps).

## 5. Automated Held-Out Scorer (`training/scorer/`)
- **`eval_suite.py`**: Locked held-out benchmark manifest across Tiers 1–4.
- **`score_checkpoint.py`**: Evaluates checkpoints on held-out maps mirroring `EVALUATION.md` scoring formulas and validating single-thread CPU latency budget ($< 40$ ms/step).

## 6. Verification Commands
```bash
# Run unit & contract tests
uv run pytest tests/test_model_and_agent.py -v
uv run pytest tests/test_mapgen_and_rewards.py -v

# Score a checkpoint against held-out maps
uv run python training/scorer/score_checkpoint.py --checkpoint submission/weights/best_agent.pt

# Stress-test 500 generated maps across Tiers 1-4 in DoomEnv
uv run python -m training.mapgen.verify_maps --count 500 --steps 10
```
