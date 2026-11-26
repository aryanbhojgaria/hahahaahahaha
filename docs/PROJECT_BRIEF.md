# DevForge Track 1: Zero-Shot Ego-Centric DOOM Agent — Project Brief

> **Team:** Vedesh (RL Core & Training), Aryan (Neural Architecture & Scorer), Aman (Procedural MapGen, Rewards & PLR Curriculum), Swapnil (Training & Docs)  
> **Target:** `vizdoom==1.3.3` | `freedoom2.wad` | Skill 3 ("Hurt me plenty") | `FRAME_SKIP = 4` | `1` CPU Thread (`< 25 ms/step`)

---

## Page 1 — Core USP, Neural Architecture & Rule Compliance

### 1. Problem & Core Innovation
Standard RL agents in FPS environments fail when evaluated on **unseen random maps** because they overfit fixed `(x, y)` coordinates, wall textures, or memorized routes. Our agent is **100% coordinate-free and self-evolving**:
1. **No Hardcoded Coordinates or Fixed Routes:** Every training map is procedurally generated from `(tier, knobs, seed)` with randomized room graphs, spawn `(x, y, angle)`, textures, lighting, and item/enemy placements.
2. **Scan-Matched Egocentric Occupancy Grid + 8-Sector Value Compass (`HybridDoomNet`):** Rather than relying on raw pixels alone, our policy estimates ego-motion `(Δx, Δy, Δθ)` via 64-ray circular scan-matching, warps a `32×32` 6-channel semantic occupancy grid (`visited`, `wall`, `door`, `locked_door`, `item`, `enemy`), and fuses it with a **Circular 1D Ray Conv**, **Masked Entity Self-Attention**, **Depthwise RGB+D CNN (with `p=0.25` modality dropout)**, **GRU (`256`)**, and **Objective Memory Slots (`4×8`)**.
3. **100% Rule-Compliant (`docs/RULES.md`):** Every action in `submission/agent.py` comes from trained PyTorch weights—zero `if/else` action trees, zero `A*`/BFS pathfinding at inference, and zero `atan2` aimbots. Only observation-legal masking (`USE` door-bounce cooldown, `ATTACK` empty-ammo mask) is applied.

### 2. Inference Budget (`1` CPU Thread)
| Check / Benchmark | Measured Latency | Resource Envelope |
| :--- | :--- | :--- |
| `validate.py` (`two_rooms.wad:MAP01`, 72-step exit) | `p50 6.9 ms` / `p99 19.4 ms` | `3.8s` init, `1720 MB` RAM, `~247 MB` GPU |
| 27-Map Validation Suite (`eval_results_baseline.json`) | **`2.73 ms` mean** / `13.10 ms` p99 | Strict 1-thread CPU, `14×` under `40 ms` cap |

---

## Page 2 — Self-Evolving Training Pipeline, Benchmarks & Limitations

### 3. Five-Stage Self-Evolving Training Pipeline
1. **Extreme Domain-Randomized Procedural MapGen (`training/mapgen/`):** Watertight ZDoom UDMF (`TEXTMAP`) generator across **Tiers 1–4** (open arenas → multi-room doors → keycard backtracking → multi-lock gauntlets) with 6 extreme scenario presets (`balanced`, `claustrophobic_ambush`, `open_arena`, `multi_branch_maze`, `scarce_ammo`, `dense_waves`) and generation-time BFS solvability verification.
2. **Coordinate-Free Dense Reward Shaper (`training/rewards.py`):** Uses `env.privileged_state()` and `env.sector_polygons()` dynamically on any map: exit (`+15 + 3*time_frac`), unique sector discovery (`+0.40`), keycard (`+4.0`), door open (`+0.60`), locked door unlock (`+5.0`), kill (`+1.0`), pickup (`+0.15`), minus damage (`-0.02/HP`), death (`-4.0`), door-bounce (`-0.30`), and wall-scrape (`-0.02`).
3. **Self-Evolving PLR Curriculum (`training/curriculum_plr.py`):** Maintains `256` active training seeds (`[100_000, 999_999]`, strictly disjoint from held-out `[1_000, 9_999]`) prioritized by `mean |GAE|` + staleness bonus, paired with `scaffolding_alpha` (`1.0 → 0.0` over `12M` steps).
4. **Recurrent PPO + RND Curiosity + Auxiliary Bearing Heads:** Truncated BPTT (`L=32`) with `done` hidden-state masking and RND intrinsic exploration.
5. **Mandatory Vanilla Fine-Tune:** Final `2–4h` fine-tune with `scaffolding_alpha = 0.0` under strict vanilla `SKILL = 3` rules.

### 4. Validation Benchmark & Disjoint Map Splits (Local Proxy Score)
Three strictly disjoint map splits (verified by seed & SHA-256): **Training** (`seeds 100_000–999_999`), **Validation** (`training/scorer/eval_suite.json`: 24 procedural WADs `1001–1024` + 3 `freedoom2.wad` maps `MAP01–03`), and **Final Test** (`training/scorer/final_test_suite.json`: 16 procedural WADs `50001–50016` + 2 `freedoom2.wad` maps `MAP04–05`, never used for training or selection). Scores below use our **local proxy formula** (official organizer weights are hidden). Note: in `plr` mode the LLM coach updates PLR knobs; in `staged` mode it logs advisory telemetry only.

| Configuration / Checkpoint (27-Map Validation Suite) | Local Proxy Score | Exit Rate | Tier 1–4 Avg | Mean `act()` ms |
| :--- | :---: | :---: | :---: | :---: |
| Random Baseline (`init_baseline`) | `0.00` | `0.0%` | `0.0` | `10.0 ms` |
| `HybridDoomNet` + PLR + RND (`806.9k` steps) | `504.13` | `3.7%` | `19.2 / 14.0 / 18.3 / 23.3` | `2.22 ms` |
| **`HybridDoomNet` + Teacher→DAgger→BC-KL PPO (submitted, `eval_results_baseline.json`)** | **`2,076.24`** | **`25.9%`** | **`69.3 / 70.4 / 24.5 / 142.3`** | **`2.73 ms`** |

### 5. Honest Limitations & Future Work
- **Scan-Matching Drift:** Rapid multi-tic wall collisions can introduce minor `Δθ` drift in the `32×32` local grid, partially mitigated by exponential decay (`0.992`) and GRU hidden state.
- **Long-Horizon Key Backtracking:** Tier 4 multi-lock gauntlets with scarce ammo remain sensitive to early exploration entropy; future work includes hierarchical subgoal options over the objective memory slots.
