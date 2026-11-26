# Product Requirements Document (PRD) — Track 1: Zero-Shot DOOM Agent

## 1. Problem & Opportunity

Standard reinforcement learning agents in 3D FPS games memorize pixel sequences on a single training map and fail immediately when walls move, textures change, or doors require backtracking for keycards. Track 1 (*"Can It Play DOOM?"*) evaluates autonomous agents **zero-shot on unseen maps** across four escalating tiers: Combat & Survival, Multi-Room Exploration, Long-Horizon Keycard Progression, and Full Gauntlet.

Because scoring is **non-binary** (weighted sum of Exit, Key/Door Progression, Unique Sector Exploration, and Combat) and manual code/replay reviews shortlist the **Top 2 finalists**, a team that pairs **custom ego-spatial neural memory** with **randomized procedural map training** and **human-demo warm-starting** has a decisive edge over off-the-shelf pixel RL baselines.

---

## 2. Primary Goal & Stakeholders

* **Goal:** Finish **#1 on the Automated Leaderboard** (advancing safely in Top 10), pass the **Top 4–5 Manual Code & Replay Audit** with zero rule flags, and secure **Top 2 in Desk Review** to win **Round 3 Finals**.
* **Target Evaluators:**
  1. **Stage 1 Automated Scorer:** Headless sandboxed evaluator testing unseen maps, timeouts (`40 ms/step`), memory caps, and partial-progress metrics.
  2. **Stage 2 Desk Reviewers:** Technical organizers running live demos, inspecting `training/` & replays, and grilling the team on architecture and RL decisions.
  3. **Round 3 Final Judges:** Panel rewarding genuine architectural innovation (bespoke spatial memory, curiosity, procedural generalization) and live presentation clarity.

---

## 3. Target Success Metrics (Measured on Held-Out Local Scorer)

| Metric | Target Threshold (by H20 Lock) | Why It Matters |
| :--- | :--- | :--- |
| **Tier 1 (Combat & Survival)** | `≥ 85%` episode survival/completion, `≥ 80%` enemy kill ratio | Secures reliable foundational points and proves combat competence. |
| **Tier 2 (Multi-Room Exploration)** | `≥ 75%` unique sector coverage, `≥ 60%` exit rate, `0` door-bounce traps | Separates spatial explorers from corner-spinning pixel bots. |
| **Tier 3 (Keycard → Locked Door → Exit)** | `≥ 45%` keycard pickup + door unlock rate, `≥ 25%` full exit rate | High tier multiplier; almost no standard PPO bots score progression here. |
| **Tier 4 (Full Gauntlet)** | Positive net combat + `≥ 50%` sector coverage + opportunistic exits | Highest tier multiplier; surviving ambushes while exploring maximizes partial points. |
| **Single-Thread `act()` Latency** | Mean **`< 20 ms`**, p99 **`< 35 ms`** (`torch.set_num_threads(1)`) | Safe headroom below official `40 ms/step` compute budget and `5 s` per-call cap. |
| **Memory & Repo Footprint** | Peak VRAM `< 1.5 GB`, RAM `< 4 GB`, Repo + LFS `< 250 MB` (`≤ 1 GB` cap) | Passes both draft (`6 GB`) and strict (`2 GB`) sandbox configurations effortlessly. |
| **Validator & Test Pass Rate** | `100%` pass on `python validate.py --docker` and `pytest` | Eliminates silent 0-score container or schema failures. |

---

## 4. Functional Requirements

| ID | Requirement | Owner | Priority |
| :--- | :--- | :--- | :--- |
| **FR-1** | **Contract-Compliant `DoomAgent` (`submission/agent.py`):** Implement `__init__(obs_space, act_space, device)`, `reset()`, and `act(obs) -> MultiBinary(9)` loading LFS weights from `submission/weights/` with zero network/write calls. | Aryan | P0 |
| **FR-2** | **Hybrid Sensory Encoder (`training/model.py`):** Fuse circular 1D CNN over 64 360° rays (+ hit-type embeddings), permutation-invariant masked attention over `≤ 32` visible entities, vitals MLP, and compact `64×48` RGB+depth CNN trained with RGB modality dropout. | Aryan | P0 |
| **FR-3** | **360° Scan-Matching Egocentric Grid + Objective Memory (`odometry_grid.py`, `model.py`):** Estimate yaw `Δθ` from consecutive 360° ray scans + turn history, warp a `32×32` 6-channel semantic occupancy grid using ego velocities, and maintain GRU + 4 explicit Objective Memory Slots (key held, locked door bearing, exit bearing, breadcrumb). | Aryan | P0 |
| **FR-4** | **Observation-Legal Action Masking (`preprocessing.py`):** Prevent door-reversing `USE` spam via a short observation-conditioned cooldown mask after `USE` button-down, and mask `ATTACK` when non-melee selected weapon ammo is zero. | Aryan | P0 |
| **FR-5** | **Parallel `doom_env` Rollout Engine (`training/envs.py`):** Multi-worker subprocess wrapper running `make_env` across 12–20 CPU workers on OMEN, supporting procedural WAD hot-reloading and training-only scaffolding toggles. | Vedesh | P0 |
| **FR-6** | **Custom Recurrent PPO + Curiosity (`ppo.py`, `rnd.py`):** PyTorch recurrent PPO from scratch (GAE, truncated BPTT with episode reset masking, 9 Bernoulli heads, auxiliary ray/proximity losses, RND + grid-novelty intrinsic bonus). | Vedesh | P0 |
| **FR-7** | **Human-Demo Recorder & BC Warm-Start (`bc/record_demos.py`, `bc/train_bc.py`):** Record `1–2 h` of human play on `doom_env`, pretrain the policy via supervised BC, and anchor early PPO updates with a decaying KL penalty (`β: 0.2 → 0.0`). | Swapnil + Vedesh | P1 |
| **FR-8** | **4-Tier Procedural UDMF Map Generator (`mapgen/`):** Generate solvable, node-built `.wad` levels across Tiers 1–4 (`< 512` sectors, BFS-verified inside generator) plus curated non-overlapping train/held-out WAD splits. | Aman | P0 |
| **FR-9** | **Dense Privileged Reward Shaper & PLR (`rewards.py`, `curriculum_plr.py`):** Shape training rewards from `privileged_state()` & `sector_polygons()`, prioritize high-learning-potential map seeds via PLR, anneal scaffolding to zero, and finish with a vanilla-only fine-tune. | Aman + Vedesh | P0 |
| **FR-10** | **Automated Held-Out Scorer & Brief Package (`scorer/`, `Brief.pdf`):** Continuous LOQ checkpoint evaluator mirroring `EVALUATION.md` on unseen maps, auto-promoting best Docker-validated weights, paired with a 2-page PDF brief and demo replays. | Aryan + Swapnil | P0 |

---

## 5. Non-Functional Requirements

* **NFR-1 (Strict Single-Thread Determinism & Speed):** `DoomAgent` enforces `torch.set_num_threads(1)` on CPU paths and completes 1,050 steps (a full 4,200-tick map) in `< 25 s` wall-clock agent time (budget: `42 s`).
* **NFR-2 (Zero Extra Runtime Dependencies):** `submission/requirements.txt` remains empty (or minimal pinned pure-Python wheels if strictly unavoidable) so container build/startup cannot fail.
* **NFR-3 (Graceful Fallback Architecture):** Setting `config.use_grid = False` or `config.use_rgb = False` cleanly bypasses the grid or RGB branch without breaking checkpoint loading or `agent.py` interfaces.
* **NFR-4 (Strict Train / Eval Separation):** Held-out map seeds, `freedoom2` benchmark maps, and held-out community WADs in `eval_suite.json` are mathematically disjoint from training seeds/files.
* **NFR-5 (Audit-Ready Code Quality):** Clean module boundaries, type hints, docstrings explaining math, and unit tests in `tests/` so manual code reviewers verify originality in minutes.

---

## 6. Out of Scope

* Training purely from high-resolution RGB pixels without spatial telemetry.
* Classical graph planners (`A*`, Dijkstra, Frontier BFS) or hardcoded combat state machines inside `submission/`.
* Multi-node distributed cloud clusters (unless both OMEN and LOQ suffer hardware failure).
* Custom C++ modifications to ViZDoom binaries (all training scaffolding is done via Python wrappers, WAD flags, or ACS).

---

## 7. Milestone Gates Summary

* **H2:** Repo live from template; interfaces frozen; `validate.py --docker` green; `envs.py` > 1,000 SPS.
* **H4:** Custom PPO smoke test learning on `two_rooms.wad` + MapGen v0; local scorer running on LOQ.
* **H6:** Swapnil onboarded; 1–2 h human demos recorded; BC warm-start beats random baseline on Tiers 1–2.
* **H8:** Long Training Run #1 active on OMEN (BC-init + PPO + PLR + RND); LOQ judge posting to W&B.
* **H12:** Scan-matched Occupancy Grid + Objective Slots integrated; ablation launched on LOQ.
* **H16 (Feature Freeze):** Architecture locked; Tier 1–3 progression verified on held-out maps.
* **H20 (Checkpoint Lock):** Final vanilla-only fine-tune completed, scored, Docker-validated, and promoted via Git LFS.
* **H23.5 (Submission Cutoff):** Code cleaned, 2-Page Brief PDF exported, replays captured, repo public & submitted.
