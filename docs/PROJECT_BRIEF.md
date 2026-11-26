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
| Component | Mean CPU Latency |
| :--- | :--- |
| Preprocessing + 64-ray scan-match + `cv2.warpAffine` | `~0.9 ms` |
| `HybridDoomNet` (Encoders + GRU + Slots + Heads) | `~4.6 ms` |
| Action mask + deterministic seeded sampling | `~0.2 ms` |
| **Total `DoomAgent.act()`** | **`~5.7 ms`** *(7× under `40 ms` cap)* |

---

## Page 2 — Self-Evolving Training Pipeline, Benchmarks & Limitations

### 3. Five-Stage Self-Evolving Training Pipeline
1. **Extreme Domain-Randomized Procedural MapGen (`training/mapgen/`):** Watertight ZDoom UDMF (`TEXTMAP`) generator across **Tiers 1–4** (open arenas → multi-room doors → keycard backtracking → multi-lock gauntlets) with 6 extreme scenario presets (`balanced`, `claustrophobic_ambush`, `open_arena`, `multi_branch_maze`, `scarce_ammo`, `dense_waves`) and generation-time BFS solvability verification.
2. **Coordinate-Free Dense Reward Shaper (`training/rewards.py`):** Uses `env.privileged_state()` and `env.sector_polygons()` dynamically on any map: exit (`+15 + 3*time_frac`), unique sector discovery (`+0.40`), keycard (`+4.0`), door open (`+0.60`), locked door unlock (`+5.0`), kill (`+1.0`), pickup (`+0.15`), minus damage (`-0.02/HP`), death (`-4.0`), door-bounce (`-0.30`), and wall-scrape (`-0.02`).
3. **Self-Evolving PLR Curriculum (`training/curriculum_plr.py`):** Maintains `256` active training seeds (`[100_000, 999_999]`, strictly disjoint from held-out `[1_000, 9_999]`) prioritized by `mean |GAE|` + staleness bonus, paired with `scaffolding_alpha` (`1.0 → 0.0` over `12M` steps).
4. **Recurrent PPO + RND Curiosity + Auxiliary Bearing Heads:** Truncated BPTT (`L=32`) with `done` hidden-state masking and RND intrinsic exploration.
5. **Mandatory Vanilla Fine-Tune:** Final `2–4h` fine-tune with `scaffolding_alpha = 0.0` under strict vanilla `SKILL = 3` rules.

### 4. Held-Out Benchmark, Temperature Ablation & Per-Tier Diagnostics
Evaluated on `training/scorer/eval_suite.json` (24 disjoint procedural WADs + 3 `freedoom2.wad` maps) under 1-thread CPU enforcement:

| Configuration / Checkpoint | Total Score | Exit Rate | Tier 1–4 Avg | Mean `act()` ms |
| :--- | :---: | :---: | :---: | :---: |
| Random Baseline (`init_baseline`) | `0.00` | `0.0%` | `0.0` | `10.0 ms` |
| `HybridDoomNet` + PLR + RND (`77.8k` steps) | `344.28` | `3.7%` | `19.0 / 6.2 / 11.7 / 15.0` | `3.28 ms` |
| `HybridDoomNet` + PLR + RND + LLM Coach (`557k` steps) | `386.14` | `0.0%` | `7.7 / 16.9 / 6.6 / 25.1` | `2.37 ms` |
| `HybridDoomNet` + PLR + RND + LLM Coach (`806.9k` steps) | `504.13` | `3.7%` | `19.2 / 14.0 / 18.3 / 23.3` | `2.22 ms` |
| **Staged Curriculum (`377.9k` steps, `tau=0.85`, submitted)** | **`1440.45`** | **`14.8%`** | **`39.7 / 42.4 / 63.4 / 66.0`** | **`5.66 ms`** |

- **Sampling Temperature (`tau`) Ablation on `377.9k` Checkpoint:** `tau=0.50` (`1217.46`, `7.4%` exit), `tau=0.65` (`1050.80`, `3.7%` exit), **`tau=0.85` (`1440.45`, `14.8%` exit, `56` kills)**, `tau=1.00` (`1085.30`, `11.1%` exit).
- **Diagnostic Breakdown (`eval_results_diagnostics.json`):** `56` total kills, `35` doors opened, `7` keycards collected, `46.8%` mean sector exploration, only `3/27` timeouts (`20` deaths / `15` early deaths).

### 5. Honest Limitations & Future Work
- **Scan-Matching Drift:** Rapid multi-tic wall collisions can introduce minor `Δθ` drift in the `32×32` local grid, partially mitigated by exponential decay (`0.992`) and GRU hidden state.
- **Early Combat Survival & Key-to-Lock Backtracking:** Diagnostic breakdown shows `15` early deaths and `0/7` collected keycards converted into unlocked doors on Tiers 3–4; continued Stage 3–4 vanilla fine-tuning directly targets this bottleneck.
