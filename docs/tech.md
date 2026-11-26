# DevForge Track 1 ("Can It Play DOOM?") — Complete Technical Reference & Judge Q&A

> **Project:** Zero-Shot Ego-Centric Autonomous DOOM Agent (`HybridDoomNet`)  
> **Engine:** ViZDoom `1.3.3` | `freedoom2.wad` | Skill 3 (*"Hurt me plenty"*) | `FRAME_SKIP = 4` (4 ticks/decision)  
> **Latest Verified Checkpoint (`6546ef1`):** **`2126.58`** held-out score across 27 unseen maps (`22.2%` exit rate, `83%` Tier 1 exits, `113` kills, `4.28 ms` mean `act()`)  
> **Docker Sandbox (`validate.py --docker`):** **`PASSED`** (`0` warnings, `9.4 ms` p50, `1556 MB` RAM, `~127 MB` VRAM, repo `8.9 MB / 1.0 GB`)

---

## 1. Core Problem & Unique Selling Proposition (USP)

Standard FPS RL agents overfit training maps by memorizing global `(x, y)` coordinates, room textures, or fixed turn sequences — causing them to spin in corners on unseen evaluation WADs.

Our **3-part USP** solves zero-shot generalization under strict partial observability (`1` CPU thread, `< 40 ms/step`):

1. **100% Coordinate-Free Egocentric Spatial Memory (`OdomGridState`):**
   - Estimates ego-yaw delta `Δθ` purely from **64-ray circular scan-matching** (no compass/world angle), warps a **`6 × 32 × 32` egocentric semantic occupancy grid** via `cv2.warpAffine`, and splats current ray hits (`visited`, `wall`, `door`, `locked_door`, `item`, `enemy`).
2. **Multi-Modal `HybridDoomNet` (`868,403` parameters):**
   - Fuses **Circular 1D Ray Conv**, **Masked Entity Self-Attention**, **Depthwise RGB+Depth CNN** (trained with `rgb_dropout_p = 0.20–0.25` so spatial navigation never relies solely on wall textures), **Vitals MLP**, **256-dim GRU**, and **4 × 8 Gated Objective Memory Slots**.
3. **Privileged Teacher DAgger → BC-KL-Anchored Recurrent PPO:**
   - Pure RL stalled on sparse multi-room/keycard rewards (`3.7%` exit). We broke through by pairing a **5-stage progressive curriculum** with **training-only** Privileged Teacher DAgger distillation followed by **BC-KL-anchored Recurrent PPO** (`ppo_from_dagger.yaml`), preserving valid movement/combat while optimizing long-horizon survival and exits.

---

## 2. Observation Preprocessing & Featurization (`training/preprocessing.py`)

`submission/agent.py` and all training loops share the exact same `preprocess()` and `update_action_state()` functions:

| Input Stream | Raw Space (`doom_env`) | Preprocessed Tensor | Normalization / Design |
| :--- | :--- | :--- | :--- |
| **Rays** | `distance (64,)`, `hit_type (64,)` (`RAY_RANGE=1024`) | `(B, 8, 64)` `float32` | `distance` in `[0, 1]` + `cos/sin` angular encoding or one-hot 6 hit-types (`none, wall, door, locked_door, enemy, item`) |
| **Entities** | `features (32, 5)`, `entity_type (32,)`, `mask (32,)` | `(B, 32, 5)`, `(B, 32)`, `(B, 32)` | Only FOV `90°` + LOS entities; NaN-safe masked |
| **Vitals** | `vitals (17,)` | `(B, 36)` `float32` | Scaled `health/100`, `armor/100`, weapon/ammo ratios, `6` key flags, ego velocities `vel_fwd/right/up`, plus previous action `(9,)` & door-cooldown flags |
| **RGB+Depth** | `rgb (120, 160, 3)`, `depth (120, 160)` | `(B, 4, 48, 64)` `float32` | Downsampled & normalized to `[0, 1]` |
| **Ego Grid** | Computed from `rays` + `vitals[14:16]` + `Δθ` | `(B, 6, 32, 32)` `float32` | `CELL_SIZE = 48.0` units; exponential decay (`0.998` for `visited`/`locked_door`, `0.992` for others) reduces scan-matching drift |
| **Action Mask** | Derived from `vitals` & `OdomGridState` | `(B, 9)` `bool` | Legal observation masking only (see §4) |

### How 64-Ray Scan-Matching Works (`_estimate_dtheta`)
1. Uses previous turn action `(TURN_LEFT - TURN_RIGHT)` as `k_prior`.
2. Filters out max-range misses (`>= 0.98`) and moving enemies (`hit_type == 4`) so only static geometry is matched.
3. Evaluates candidate circular shifts `k ∈ [k_prior - 4, k_prior + 4]` by MSE between `ray_dist` and `np.roll(prev_ray_dist, -k)`, plus a quadratic prior penalty.
4. Applies 3-point parabolic sub-bin refinement for smooth sub-degree `Δθ` estimation.

---

## 3. Neural Architecture: `HybridDoomNet` (`training/model.py`)

Total parameters: **`868,403`** (less than half of the `2,000,000` limit).

1. **Encoders (Parallel Feature Extraction → `336` dims):**
   - `RayCircularEncoder`: 3× `Conv1d` with **`padding_mode="circular"`** (`8 → 32 → 64 → 64`, kernel `5, 3, 3`) + `AdaptiveAvgPool1d(8)` + `Linear(512 → 96)` (`SiLU`). Circular padding guarantees `360°` wrap-around continuity between ray `63` and ray `0`.
   - `EntitySetEncoder`: `Embedding(12, 16)` + `Linear(21 → 64)` → `LayerNorm` → 4-head `MultiheadAttention` with `key_padding_mask` (explicitly handles `0` visible entities without softmax `NaN`) → masked `mean + max` pool → `Linear(128 → 64)`.
   - `VisualEncoder`: 3× `DepthwiseSeparableConv2d` (`4 → 16 → 32 → 48`, stride `2`) → global `avg + max` pool → `Linear(96 → 64)`. Zeroed with probability `rgb_dropout_p` during training.
   - `GridEncoder`: 3× `Conv2d` (`6 → 24 → 48 → 64`, stride `2`) → `Linear(1024 → 64)`.
   - `vitals_enc`: `Linear(36 → 64) → SiLU → Linear(64 → 48) → SiLU`.
2. **Recurrent Spatial-Temporal Core (`288` dims):**
   - `Fusion`: `LayerNorm(336) → Linear(336 → 256) → SiLU`.
   - `GRU(256, 256)`: Hidden state `gru_h (1, B, 256)` explicitly masked to `0` on `done[t] == 1` (`RecurrentState.mask_done()`).
   - **Gated Objective Memory Slots (`4 × 8 = 32` dims):** `slot_write(gru_h)` blended via sigmoid `slot_gate([gru_h, prev_slot])` to retain long-horizon keycard/door state across hundreds of ticks.
3. **Output Heads:**
   - **Inference Heads:** `policy_head (288 → 128 → 9)` (MultiBinary Bernoulli logits) and `value_head (288 → 128 → 1)`.
   - **Training-Only Auxiliary Heads:** `aux_next_rays (64)`, `aux_value_compass (8-sector ego reward attribution)`, `aux_door_prox (3)`, `aux_key_prox (3)`, `aux_exit_prox (3)`. Removed/ignored at inference for speed.

---

## 4. Rule Compliance & Inference Pipeline (`submission/agent.py`)

Strictly compliant with `docs/RULES.md` and `docs/CONTRACT.md`:
- **No Rule Trees / No A* / No Aimbots:** Every action logit comes directly from `HybridDoomNet.forward()`. Zero `if/else` action heuristics, zero BFS/Dijkstra at inference, zero `atan2` enemy aiming, zero `privileged_state()` access.
- **Observation-Legal Action Masking (`compute_action_mask`):**
  1. `USE` (`idx 7`) cooldown mask: prevents holding `USE` across consecutive steps or bouncing a moving door (`docs/CONTRACT.md §3`).
  2. `ATTACK` (`idx 6`) empty-ammo mask: disables firing when `selected_weapon > 1` and `selected_weapon_ammo <= 0`.
- **Calibrated Sampling (`tau = 0.85`):**
  - Low-temperature Bernoulli sampling (`sigmoid(logits / 0.85)`) with deterministic seeded RNG (`default_rng(0)` reset per map) and opposite-axis mutex stabilization (`FWD/BAK`, `LEFT/RIGHT`, `TURN_L/TURN_R` keeps higher logit if both fire).
  - **Empirical `tau` Ablation (27 held-out maps on `377.9k` ckpt):** `tau=0.50` (`1217.46`), `tau=0.65` (`1050.80`), **`tau=0.85` (`1440.45`, `14.8%` exit)**, `tau=1.00` (`1085.30`).
- **Crash & NaN Safety:** `np.nan_to_num` on logits/observations + `torch.set_num_threads(1)` enforced on CPU.

---

## 5. Training Pipeline, MapGen & Breakthrough History

### A. Procedural UDMF Map Generator (`training/mapgen/tiers.py`)
- Pure-Python ZDoom UDMF (`TEXTMAP`) generator (< `2048` sectors) with randomized global `(origin_x, origin_y) ∈ [-2048, 2048]`, `90°` rotations, `cell_size ∈ [160, 384]`, corridor widths `64..128`, textures, lighting, and 7 archetypes (`open_arena`, `tight_ambush`, `branching_maze`, `scarce_ammo`, `dense_waves`, `multi_room_doors`, `keycard_progression`).
- Generation-time **BFS solvability verification** (`start → keycard(s) → locked door(s) → exit`) guarantees every generated map is completable.
- **Strict Train/Eval Split:** Training seeds `[100_000, 999_999]`; held-out evaluation seeds `[1_000, 9_999]` (24 procedural WADs + 3 `freedoom2.wad` maps = 27 maps in `eval_suite.json`).

### B. Dense Coordinate-Free Reward Shaper (`training/rewards.py`)
Uses only `env.privileged_state()` deltas and `env.sector_polygons()` (zero world XYZ assumptions):
- `exit_base: +15.0` (+ up to `+3.0` speed bonus) | `locked_door_unlock: +5.0` | `key_pickup: +4.0` | `door_open: +1.2` | `kill: +1.0` | `new_sector: +0.40` | `new_cell: +0.06` | `item_pickup: +0.15`
- Penalties: `death: -4.0` | `door_reversal_bounce: -0.30` | `damage_per_hp: -0.02` | `stuck_wall_scrape: -0.02` | `wasted_shot: -0.015` | `loop_penalty: -0.015` | `step_cost: -0.002`

### C. Why Earlier Training Stalled & How We Fixed It

| Iteration / Method | Held-Out Score (27 Maps) | Exit Rate | Kills | Key Insight |
| :--- | :---: | :---: | :---: | :--- |
| Random Baseline | `0.00` | `0.0%` | `0` | Untrained |
| Hard-Biased PLR + PPO (`77.8k` steps) | `344.28` | `3.7%` | — | `tier_weights: [1, 1.5, 2.5, 3]` threw agent into 75% Tier 3–4 maps immediately; `temperature=0.1` collapsed replay |
| PLR + RND + LLM Coach (`806.9k` steps) | `504.13` | `3.7%` | — | Exploration improved, but sparse exits & spinning habits persisted |
| **5-Stage Progressive Curriculum (`377.9k`)** | **`1440.45`** | **`14.8%`** | **`56`** | `single_map → easy → medium → hard → random` gated by rolling exit rate (`80%→60%→45%→30%`) |
| **Privileged Teacher DAgger → BC-KL PPO (`6546ef1`)** | **`2126.58`** | **`22.2%`** | **`113`** | Teacher distillation cured early deaths (`20 → 10`) & key-to-lock progression (`0 → 5` locks unlocked), then PPO fine-tuned on vanilla rules |

---

## 6. Verification, Latency & Memory Benchmarks

All tests verified after final edits on commit `13a757c`:

| Check / Environment | Result | Init | `p50` / `p99` Latency | Budget Used | RAM / VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Docker Sandbox (`validate.py --docker`, `cuda`)** | **PASSED (0 warn)** | `5.7s` | `9.4 ms` / `16.8 ms` | `2.3s / 21.0s` (`84` steps, exit) | `1556 MB` / `~127 MB` |
| **Tier 4 Gauntlet (`t4_seed1004.wad`, `cuda`, 1 CPU)** | **PASSED (0 warn)** | `2.7s` | `12.2 ms` / `18.6 ms` | `14.5s / 42.0s` (`1050` steps) | `1330 MB` / `~135 MB` |
| **1-Thread CPU-Only (`two_rooms.wad`, `cpu`)** | **PASSED (0 warn)** | `2.0s` | `8.0 ms` / `14.0 ms` | `0.7s / 21.0s` (`84` steps, exit) | `601 MB` / `N/A` |
| **Unit Tests (`tests/test_scorer_and_agent_guard.py`)** | **4/4 PASSED** | — | — | `1.95s` | — |

- **Per-Tier Held-Out Breakdown (`eval_results_diagnostics.json`):**
  - **Tier 1:** `5/6` exits (`83%`), `64.99` avg score, `95.8%` exploration, `3.83` avg kills
  - **Tier 2:** `1/7` exits (`14%`), `56.03` avg score, `59.9%` exploration, `4.29` avg kills
  - **Tier 3:** `0/7` exits, `69.65` avg score, `37.9%` exploration, `2.86` avg kills (`5` locks unlocked across suite)
  - **Tier 4:** `0/7` exits, `122.41` avg score, `23.2%` exploration, `5.71` avg kills

---

## 7. Judge Q&A Cheat Sheet

1. **"How do we know your agent isn't using `A*`, `atan2`, or `if/else` rules at evaluation?"**
   - Inspect `submission/agent.py`: `act()` runs `preprocess()` → `self.model(inputs, self.recurrent_state)` → `sigmoid(logits / tau)` → `action_mask` (`USE` cooldown & empty-ammo `ATTACK`). Zero `env.privileged_state()`, zero coordinate lookups, zero heuristic action trees.
2. **"Isn't using `PrivilegedTeacherBot` in DAgger against the rules?"**
   - `docs/RULES.md §2` explicitly permits any training-time scaffolding, privileged state, or teacher scripts (`training/bc/` is training-only). At evaluation time, `submission/agent.py` receives only the standard `doom_env` observations under vanilla Skill 3 rules.
3. **"How do you prevent overfitting to `two_rooms.wad` or training seeds?"**
   - Every training map uses random origin offsets `[-2048, 2048]`, `90°` rotations, randomized textures/lighting, and seeds `[100_000, 999_999]` — completely disjoint from `eval_suite.json` seeds `[1_000, 9_999]` and `freedoom2.wad` maps.
4. **"What happens if `rgb` textures look completely different in hidden maps?"**
   - `VisualEncoder` is trained with `rgb_dropout_p = 0.20–0.25`, forcing `HybridDoomNet` to rely primarily on geometry-invariant `RayCircularEncoder` (`64` rays), `EntitySetEncoder`, and `GridEncoder`.
5. **"How does the agent remember a locked door after picking up a keycard across the map?"**
   - Two mechanisms: (a) the `32×32` egocentric grid retains `CH_LOCKED_DOOR` with slow decay (`0.998`), and (b) the `GRU(256)` + `4×8` gated Objective Memory Slots carry persistent latent state across the episode (unrolled with truncated BPTT `L=32`).
6. **"Why `tau = 0.85` instead of greedy `argmax`/`> 0.5`?"**
   - On symmetric corners or doorways, deterministic thresholding can trap a reactive policy in a 2-step limit cycle. Our 27-map ablation proved `tau = 0.85` achieves **`1440.45`** vs `1050.80` (`tau=0.65`) and `1085.30` (`tau=1.00`).
7. **"What is your current bottleneck if you had another 12 hours?"**
   - Tier 3–4 long-horizon backtracking after unlocking doors (`37.9%` Tier 3 exploration, `5` timeouts on Tier 3) and Tier 4 early ambush survival (`5` deaths). Longer BC-KL-anchored Stage 4 PPO fine-tuning + longer BPTT horizon (`64`) directly targets this.
