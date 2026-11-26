# DevForge "Can It Play DOOM?" — Complete Technical Guide (Simple & Judge-Ready)

> **Team:** Aryan (features, spatial mapping, evaluation & systems) + Vedesh (RL training, DAgger & curriculum)  
> **Latest Verified Score:** **3,122.37** on 27 unseen held-out maps (`22.2%` zero-shot exit rate)  
> **Inference Speed:** **3.02 ms** mean / **13.03 ms** p99 on **1 CPU thread** (budget: `40.0 ms`)  
> **Model Size:** **868,403** parameters (budget: `< 2,000,000`)  
> **Docker Sandbox (`validate.py`):** ✅ **PASSED**

---

## 🎯 1. What Is This Project? (From Basics)

We built an autonomous AI agent that plays **3D DOOM** completely on its own — it explores unknown rooms, fights enemies, collects keys, unlocks doors, and finds the exit — **without memorizing any map** and **without ever seeing GPS/world coordinates**.

### How We Built It From Scratch (Step-by-Step)
1. **Step 1 — Understand the Rules & Constraints:**  
   At test time, the agent gets **no `(x, y, z)` coordinates**, **no `A*` pathfinding**, **no `atan2` aimbots**, and **no `if/else` rule trees**. It must run on **1 CPU thread**, under **40 ms per decision**, and stay under **2M parameters**.
2. **Step 2 — Build Coordinate-Free Senses & Mental Map (`training/preprocessing.py`, `training/odometry_grid.py`):**  
   Instead of raw pixels alone, we built a preprocessing pipeline that converts raw ViZDoom observations into 5 clean signals — including a **self-drawn 32×32 egocentric map** built purely by comparing 64 radar rays frame-to-frame (scan-matching).
3. **Step 3 — Design the Brain (`HybridDoomNet` in `training/model.py`):**  
   We designed a compact `868K`-parameter multi-modal neural network that fuses all 5 senses into a **256-dim GRU (working memory)** + **4×8 Goal Memory Slots (long-term sticky notes)** to output 9 binary controller buttons.
4. **Step 4 — Build Our Own Procedural Map Factory (`training/mapgen/`):**  
   Training on 1–2 fixed maps causes memorization. So we wrote a pure-Python ZDoom UDMF map generator that creates infinite random maps across 4 difficulty tiers — and checks with BFS at generation time that every map is actually solvable.
5. **Step 5 — Build Fair, Coordinate-Free Rewards (`training/rewards.py`):**  
   During training only, we reward exploration (new sectors), combat, key pickups, unlocking doors, and exiting — while penalizing bad habits (wall-scraping, bouncing doors, taking damage, dying, looping).
6. **Step 6 — Evolve the Training Pipeline (`training/ppo.py`, `training/bc/`):**  
   - Pure RL failed (`344` score, `3.7%` exits) because keycard/exit rewards are too sparse.
   - Adding a **5-stage progressive curriculum** jumped the score to **`1,440.45`** (`14.8%` exits).
   - Adding **Privileged Teacher DAgger → BC-KL-Anchored Recurrent PPO** broke through to **`2,126.58`** and then **`3,122.37`** (`22.2%` exits).
7. **Step 7 — Verification, Safety & Demo Tooling:**  
   We built automated unit tests (`pytest`), a checkpoint comparison gate (`compare_checkpoints.py`), Docker sandbox validation (`validate.py`), diagnostics charts, and a live 3-panel spectator/replay HUD.

---

## 🧠 2. Algorithm Used — And Why?

We use a **3-part hybrid learning pipeline**:

### A. Recurrent PPO (Proximal Policy Optimization) + GAE
- **What it is:** Our core Reinforcement Learning (RL) algorithm. PPO updates the policy safely in small clipped steps (`clip_coef = 0.15`) so training doesn't collapse, using Generalized Advantage Estimation (`gamma = 0.99`, `gae_lambda = 0.95`) and a **GRU** unrolled with truncated Backpropagation Through Time (`bptt_len = 64`).
- **Why PPO + GRU?** DOOM is *partially observable* — you can't see behind you, and you forget where a locked door was unless you have memory. A standard feed-forward network (like DQN or plain MLP PPO) fails on multi-room navigation.

### B. Privileged Teacher DAgger (Dataset Aggregation)
- **What it is:** During training *only* (explicitly allowed by `RULES.md`), a scripted `PrivilegedTeacherBot` that can see map geometry shows the student network how to navigate, fight, grab keys, and unlock doors. The student learns by Behavioral Cloning (imitating the teacher on states the student itself visits).
- **Why DAgger?** In Tier 3–4 maps, random exploration almost *never* stumbles onto `spawn → key → locked door → exit` by luck. Without a teacher, pure RL wastes millions of steps spinning in rooms. DAgger teaches the agent basic competence first.

### C. BC-KL Anchor + PLR + RND (Self-Practice Without Forgetting)
- **BC-KL Anchor (`bc_kl_beta_init = 0.035`):** When switching from Teacher DAgger back to PPO self-practice, PPO can accidentally "forget" how to aim or open doors while chasing rewards. Keeping a gentle KL-divergence tether to the DAgger checkpoint prevents catastrophic forgetting while letting PPO discover even better survival and exit strategies.
- **PLR (Prioritized Level Replay):** Automatically replays maps where the agent still has high learning potential (`mean |GAE|` + staleness) instead of wasting time on maps it already mastered.
- **RND (Random Network Distillation) + Grid Novelty:** Gives a small intrinsic curiosity bonus for visiting genuinely new visual states and new grid cells.

---

## 🏆 3. How Are We Better Than Others?

| Aspect | Typical / Naive DOOM RL Bot | Our Agent (`HybridDoomNet`) |
| :--- | :--- | :--- |
| **Position & Map** | Memorizes `(x, y)` coordinates or fails when spawned in a rotated room | **100% coordinate-free.** Builds a live `32×32` rotating egocentric map via **64-ray scan-matching** — looks identical from any angle |
| **Visual Overfitting** | Memorizes wall colors/textures (`rgb` only) and breaks on new WADs | Trained with **`rgb_dropout_p = 0.20–0.25`** + randomized textures/lighting, forcing reliance on geometry (rays + depth + grid) |
| **Memory** | Frame-stacking only (remembers ~0.5 sec) | **256-dim GRU + 4×8 Gated Objective Memory Slots** (remembers keys & locked doors across hundreds of ticks) |
| **Training Strategy** | Pure PPO from scratch (gets stuck at ~3–5% exit rate on sparse keycard maps) | **5-Stage Curriculum → Teacher DAgger → BC-KL PPO** (`22.2%` exit rate, `3,122.37` score) |
| **Speed & Size** | Heavy ResNets or slow search algorithms | **868,403 params**, **3.02 ms** mean / **13.03 ms** p99 on **1 CPU thread** (13× faster than the 40 ms limit) |
| **Evaluation Rigor** | Tested on training maps | Tested on **strictly disjoint** seeds (`1,000–9,999` vs `100,000–999,999`) + `freedoom2.wad` + `final_test_suite.json` |

---

## 🏗️ 4. Architecture — How the Brain Works

Think of `HybridDoomNet` (**868,403 parameters**) as **5 specialized senses → Fusion → 2 Memory Systems → Action & Value Heads**:

```
[1. 64 Radar Rays (8×64)]      ──▶ RayCircularEncoder (360° circular Conv1d) ──▶ 96 dims ─┐
[2. Nearby Entities (32×5)]    ──▶ EntitySetEncoder (4-head Masked Attention) ──▶ 64 dims ─┤
[3. Camera RGB+Depth (4×48×64)]──▶ VisualEncoder (Depthwise-Sep CNN)          ──▶ 64 dims ─┼─▶ LayerNorm + Linear(336→256)
[4. Self-Built Map (6×32×32)]  ──▶ GridEncoder (2D ConvNet)                   ──▶ 64 dims ─┤         │
[5. Health/Ammo/Keys (36)]     ──▶ Vitals MLP                                 ──▶ 48 dims ─┘         ▼
                                                                                            [256-dim GRU]
                                                                                                 │
                                                                                + [4×8 Goal Memory Slots (32 dims)]
                                                                                                 │
                                                                           ┌─────────────────────┴─────────────────────┐
                                                                           ▼                                           ▼
                                                                 [Policy Head → 9 Buttons]                   [Value Head → 1]
```

### Plain-English Breakdown of Each Piece
1. **`RayCircularEncoder`:** Uses `padding_mode="circular"` on 1D convolutions so ray #63 wraps seamlessly into ray #0 — no blind spot behind the agent's back.
2. **`EntitySetEncoder`:** Uses masked self-attention over up to 32 visible enemies/items/keys. Explicitly handles `0` visible entities safely without `NaN` crashes.
3. **`VisualEncoder`:** Lightweight depthwise-separable CNN over downsampled `48×64` RGB + Depth. Randomly zeroed out during training (`rgb_dropout_p`) so the agent never becomes texture-dependent.
4. **`GridEncoder` (`OdomGridState`):** Encodes the `6×32×32` local map (`visited`, `wall`, `door`, `locked_door`, `item`, `enemy`). Every step, `_estimate_dtheta` compares current vs previous static ray distances to estimate how much the agent turned, warps the grid with `cv2.warpAffine`, paints new hits, and slowly decays old cells (`0.998` for visited/locked doors, `0.992` for others).
5. **`vitals_enc`:** Encodes health, armor, weapon, ammo ratios, 6 keycard flags, local velocities, previous action, and door cooldown flags.
6. **Memory Core (`GRU` + `4×8` Objective Memory Slots):** The GRU tracks short/medium-term context, while gated memory slots act like sticky notes for long-horizon goals (e.g., *"I picked up the blue key, now go back to the locked door"*).
7. **Inference Sampling (`submission/agent.py`):**
   - Applies temperature **`tau = 0.85`** Bernoulli sampling (`sigmoid(logits / 0.85)`) with a deterministic per-map RNG (`default_rng(0)`). Why `0.85`? Greedy `argmax` gets stuck in symmetric doorway loops; `tau=0.85` beat `0.65` and `1.00` in our 27-map ablation.
   - Resolves conflicting opposite buttons (`FWD/BAK`, `LEFT/RIGHT`, `TURN_L/TURN_R`) by keeping the higher-logit action.
   - Applies **only 2 legal observation-derived masks**: `USE` cooldown (don't bounce a moving door) and `ATTACK` empty-ammo mask.

---

## ⚖️ 5. Positive Rewards & Negative Penalties (`training/rewards.py`)

Because DOOM's raw exit reward is too sparse, we use **coordinate-free delta shaping** during training (zero world XYZ assumptions):

### Positive Rewards (What We Encourage)
- **Exit reached:** `+15.0` base + up to `+3.0` speed bonus (up to `+30.0` / `+6.0` in `ppo_v2_exit_opt.yaml`)
- **Unlocking a locked door:** `+5.0` (`+12.0` in exit-opt)
- **Picking up a keycard:** `+4.0` (`+10.0` in exit-opt)
- **Opening a normal door:** `+1.2`
- **Killing an enemy:** `+1.0`
- **Discovering a new map sector / item:** `+0.40` / `+0.15`

### Negative Penalties (What We Discourage & Why)
| Penalty | Weight (Base / Exit-Opt) | Why We Added It |
| :--- | :---: | :--- |
| **Death** | `-4.00` | Teaches survival and dodging instead of rushing blindly into ambushes |
| **Damage taken (per HP)** | `-0.02 / HP` | Encourages strafing and taking cover during firefights |
| **Door reversal bounce** | `-0.30` | Stops the agent from spamming `USE` and slamming a door shut right after opening it |
| **Locked door without key** | `-0.12` *(exit-opt)* | Prevents wasting time bumping into a locked door before finding its keycard |
| **Stuck / wall scrape** | `-0.02` / `-0.10` | Penalizes holding forward while velocity is near zero against a wall |
| **Empty wall shot** | `-0.08` *(exit-opt)* | Stops wasteful blind firing into walls when no enemy is in sight |
| **Unnecessary weapon swap** | `-0.06` *(exit-opt)* | Prevents flickering between weapons mid-combat |
| **Looping / wall spin** | `-0.05` / `-0.04` *(exit-opt)* | Breaks repetitive spinning-in-place or pacing back-and-forth in the same spot |
| **Step cost** | `-0.002 / step` | Encourages direct, efficient routes to the exit |

> **Engineering Note:** We intentionally removed the naive `wasted_shot` penalty (`0.0` in `rewards.py`) because `doom_env.privileged_state()` does not expose `damage_dealt` or `hit_count` — penalizing every non-lethal shot was unfairly punishing the agent for shooting tough enemies!

---

## 📦 6. Datasets, Environments & Splits Used

Instead of a static offline dataset, our agent trains and is evaluated in **live interactive 3D simulation**:

1. **Simulation Engine:** `ViZDoom 1.3.3` running `freedoom2.wad` at **Skill 3 (*"Hurt me plenty"*)**, `FRAME_SKIP = 4` (4 game ticks per agent decision).
2. **Procedural 4-Tier UDMF Map Generator (`training/mapgen/`):**
   Generates custom ZDoom `TEXTMAP` PWADs with randomized global origin offsets (`[-2048, 2048]`), `90°` rotations, room/corridor sizes, lighting, textures, and **7 archetypes**: `open_arena`, `tight_ambush`, `branching_maze`, `scarce_ammo`, `dense_waves`, `multi_room_doors`, `keycard_progression`. Every map passes a **BFS solvability check** before being saved.
3. **Strict Disjoint Train / Validation / Test Splits (Zero Data Leakage):**
   - **Training Pool:** Procedural seeds `100,000 – 999,999` (256 active seeds managed by `PLRBuffer`).
   - **Held-Out Validation (`training/scorer/eval_suite.json`):** `27` maps — `24` procedural WADs from disjoint seeds `1,000 – 9,999` (6 per tier) + `3` unseen `freedoom2.wad` maps.
   - **Final Test Suite (`training/scorer/final_test_suite.json`):** `16` completely separate unseen maps (`seeds 50,001 – 50,016`, 4 per tier) never touched during tuning or ablations.
4. **Online DAgger Trajectories:** Rollouts generated dynamically by `PrivilegedTeacherBot` during the BC-DAgger phase (stored locally in `runs/`, never shipped in `submission/`).

---

## 📊 7. Results & Progress Timeline

### Training Evolution
| Stage / Approach | Held-Out Score (27 Maps) | Exit Rate | Key Takeaway |
| :--- | :---: | :---: | :--- |
| Random Baseline | `0.00` | `0.0%` | Untrained control |
| Hard-Biased PPO (`77.8k`) | `344.28` | `3.7%` | Thrown into hard maps too early; collapsed |
| PPO + RND + LLM Coach (`806.9k`) | `504.13` | `3.7%` | Explored more, but sparse exits & spinning remained |
| **5-Stage Curriculum (`377.9k`)** | **`1,440.45`** | **`14.8%`** | Progressive difficulty unlocked navigation |
| **Teacher DAgger → BC-KL PPO (`881k`)** | **`3,122.37`** | **`22.2%`** | Tier 1–4 avg: `51.7 / 90.2 / 114.7 / 196.9`; `3.02 ms` mean latency |

### Compliance & Speed Verification
- **Docker Sandbox (`validate.py`):** **PASSED** (0 warnings, `1556 MB` RAM, `~127 MB` VRAM, repo size well under `1 GB`).
- **Single-Thread CPU Latency:** `3.02 ms` mean / `13.03 ms` p99 (`< 40 ms` limit).
- **Automated Tests (`pytest`):** All unit & audit tests passing (`tests/test_scorer_and_agent_guard.py`, `tests/test_audit_fixes.py`).

---

## 🚀 8. Future Scope

If we had more compute and time beyond this hackathon, here is our roadmap:
1. **Grid-Guided Frontier & Goal Attention:** Add an explicit spatial attention head over the `32×32` egocentric occupancy grid so the agent can directly target *"nearest unexplored frontier"* or *"remembered locked door"* after picking up a keycard.
2. **Longer Horizon Memory (`bptt_len = 128`+):** Extend sequence unrolling and Stage 3–4 PPO fine-tuning so the agent consistently backtracks across large multi-keycard mazes (Tier 3–4).
3. **Tactical Combat & Resource Management:** Improve Tier 4 ambush survival by training explicit cover-seeking, strafe-dodging, and ammo-conservation across multiple weapons.
4. **Full Campaign & Sim-to-Real Transfer:** Scale the same coordinate-free scan-matched occupancy architecture to full multi-level `freedoom2.wad` episodes and real-world robotics / indoor drone navigation where GPS is unavailable.

---

## 🙋 9. Judge Q&A Cheat Sheet

**Q1: "How do we know your agent isn't using `A*`, coordinates, or `if/else` rules at evaluation?"**  
> Inspect `submission/agent.py`. `act()` runs `preprocess()` → `HybridDoomNet` → `sigmoid(logits / 0.85)` → 2 observation-only legal masks (`USE` cooldown + empty-ammo `ATTACK`). There is zero `privileged_state()`, zero `atan2`, zero pathfinding, and zero heuristic action trees.

**Q2: "Isn't using `PrivilegedTeacherBot` in DAgger against the rules?"**  
> No — `docs/RULES.md` explicitly allows any training-time scaffolding, privileged state, or teacher scripts. `training/bc/` is strictly training-only. At evaluation time, `submission/agent.py` receives only standard `doom_env` observations.

**Q3: "What if the hidden evaluation maps have completely different textures or layouts?"**  
> During training we randomized origin offsets, `90°` rotations, room sizes, lighting, and textures — and even dropped out the camera (`rgb_dropout_p = 0.20–0.25`) so the agent relies on geometry-invariant 64-ray radar, depth, and its rotating `32×32` occupancy grid.

**Q4: "How does it remember a locked door after walking across the map to find a key?"**  
> Two systems work together: the `32×32` egocentric grid retains `locked_door` cells with slow `0.998` decay, and the `256`-dim `GRU` + `4×8` gated Objective Memory Slots carry persistent latent state across hundreds of steps.

**Q5: "Why `tau = 0.85` instead of deterministic `> 0.5`?"**  
> Deterministic thresholding traps reactive policies in symmetric doorway/corner loops. Our 27-map ablation showed `tau = 0.85` (`1,440.45`) significantly outperforms both `0.65` (`1,050.80`) and `1.00` (`1,085.30`).

**Q6: "Why did you remove the shooting penalty (`wasted_shot`)?"**  
> Because `doom_env` only reports `kill_delta`, not `damage_dealt`. Penalizing shots whenever `kill_delta == 0` punished the agent for shooting high-HP enemies that take multiple hits. We replaced it with `empty_wall_shot` in `ppo_v2_exit_opt.yaml` which only penalizes firing into blank walls with no enemy visible.
