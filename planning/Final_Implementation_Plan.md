# Master Phase-Wise Implementation Plan (24 Hours)

---

## Phase 0: Pre-Event Readiness (Before H0 — Zero Project Code!)
* **Vedesh (OMEN):** WSL2 Ubuntu 24.04 + CUDA PyTorch + `uv` + Python 3.13 + Docker Desktop (`devforge-doom` image cached) + `zdbsp` + SSH server + Tailscale + W&B team workspace. Verify core count (`nproc`).
* **Aryan (LOQ):** Identical WSL2 Ubuntu 24.04 + CUDA PyTorch + `uv` + Docker Desktop (`devforge-doom` cached) + Tailscale + W&B login.
* **Aman (M5):** macOS `uv` + Python 3.13 + `vizdoom==1.3.3` + `omgifol` + `zdbsp`/`bsp` verified + Tailscale SSH to OMEN tested.
* **Swapnil (M3):** macOS `uv` + Python 3.13 + `vizdoom==1.3.3` + `pygame` verified + Tailscale SSH tested + 15-min Freedoom keyboard control practice.
* **Gate 0:** All 4 machines pass setup commands in `requirements.md`; any temporary `/tmp/setup-check` template clones deleted prior to H0.

---

## Phase 1: H0–H2 — Repo Init, Schema Lock & Skeleton Pipeline
* **Vedesh (OMEN):**
  1. At H0:00, click **"Use this template"** on `Reinforce-SST/devforge-doom-template` → create public repo, initialize Git LFS, protect `main`, invite Aryan, Aman, Swapnil.
  2. Create `training/configs/` & `training/envs.py` (multiprocess subprocess `doom_env` workers with `spawn`, episode auto-reset, and map-reload hook).
* **Aryan (LOQ):**
  1. Implement `training/preprocessing.py` v0 (rays, entities, vitals, `64×48` RGB+depth, legal `USE`/`ATTACK` mask; dummy zero grid placeholder).
  2. Implement `training/model.py` v0 (Ray 1D Conv + Entity Attention + Vitals MLP + RGB CNN + GRU + 9 Bernoulli heads + Critic) and wire `submission/agent.py` loading a dummy/untrained `.pt` checkpoint.
* **Aman (M5):**
  1. Implement `training/mapgen/udmf_builder.py` (or classic WAD lump writer + `zdbsp` node builder) and `tiers.py` v0 generating solvable Tier 1 arenas and Tier 2 2–3 room door maps.
  2. Assemble initial practice map pool (`two_rooms.wad`, `tests/fixtures/*.wad`, ViZDoom scenarios).
* **Swapnil (M3):** *(Not yet joined — joins at H3–H4).*
* **Merge Order:** Aryan merges `feat/aryan-model-v0` → Aman merges `feat/aman-mapgen-v0` → Vedesh merges `feat/vedesh-envs`.
* **H2 Gate:** `python validate.py --docker` passes on `submission/agent.py` with randomly initialized `HybridDoomNet`; `envs.py` steps 16 workers on OMEN at `> 1,000` steps/sec.

---

## Phase 2: H2–H4 — Custom PPO, Dense Rewards & Local Held-Out Scorer
* **Vedesh (OMEN):**
  1. Implement `training/ppo.py` (custom GAE, truncated BPTT `L=32` with `done` hidden-state masking, clipped policy/value losses, 9 Bernoulli heads) and `training/rnd.py`.
  2. Run first end-to-end PPO smoke run on `two_rooms.wad` + Aman's Tier 1–2 maps; stream metrics to W&B.
* **Aryan (LOQ):**
  1. Implement `training/scorer/eval_suite.py` & `score_checkpoint.py` mirroring `EVALUATION.md` (Exit, Progression, Sector Coverage via `sector_polygons()`, Combat, plus single-thread `act()` ms timer).
  2. Write `tests/test_model_and_agent.py` verifying shapes, reset isolation, and `torch.set_num_threads(1)` latency (`< 20 ms`).
* **Aman (M5):**
  1. Implement `training/rewards.py` using `privileged_state()` & `sector_polygons()` (dense sector discovery, kills, door open, anti-wall-scrape, anti-`USE`-bounce).
  2. Lock fixed held-out benchmark WADs (`seed ∈ [1000, 9999]`) for Aryan's `eval_suite.json`.
* **Merge Order:** Aman (`feat/aman-rewards`) → Aryan (`feat/aryan-scorer`) → Vedesh (`feat/vedesh-ppo`).
* **H4 Gate:** PPO smoke test shows rising episode return and kill/door counts on W&B within 15 minutes; `score_checkpoint.py` produces a complete JSON scorecard on LOQ.

---

## Phase 3: H4–H6 — Swapnil Joins, Human Demos, BC Warm-Start, Odometry Grid & Tiers 3–4
* **Swapnil (M3 — Onboards at H3.5–H4):**
  1. Clone repo, sync `uv`, build `training/bc/record_demos.py` (`pygame` 320×240 preview + keyboard → `MultiBinary(9)` saved as `.npz` per `Schema.md`).
  2. Record `1–1.5 hours` of clean human trajectories across Aman's Tier 1, Tier 2, and Tier 3 maps (Aman records 20–30 mins in parallel too!) and `rsync` `.npz` files to OMEN.
* **Vedesh (OMEN):**
  1. Implement `training/bc/train_bc.py` (sequence cross-entropy over human `.npz` chunks) + decaying **BC-KL regularization** (`β: 0.2 → 0.0`) inside `ppo.py`.
  2. Pretrain BC checkpoint in `< 10 min` once Swapnil syncs first batch of demos.
* **Aryan (LOQ):**
  1. Implement `training/odometry_grid.py` (360° circular ray scan-matching + `cv2.warpAffine` `32×32` 6-ch semantic occupancy grid) + Objective Memory Slots + Auxiliary loss heads (`next_rays`, `door/key bearing`).
  2. Visually sanity-check warped grids on `tests/fixtures` and benchmark single-thread latency.
* **Aman (M5):**
  1. Complete Tier 3 (keycard branch → locked door → exit backtracking) and Tier 4 (multi-lock + ambush + scarce resource gauntlet) procedural generators with internal BFS solvability assertions.
  2. Implement `training/curriculum_plr.py` (Prioritized Level Replay + scaffolding knob annealing).
* **Merge Order:** Aryan (`feat/aryan-grid-slots`) → Aman (`feat/aman-tier34-plr`) → Swapnil (`feat/swapnil-bc-recorder`) → Vedesh (`feat/vedesh-bc-ppo`).
* **H6 Gate:** BC warm-started model clearly opens doors and shoots enemies on held-out Tiers 1–2 (`score > random` by ≥ 2.5×); Tiers 1–4 procedural WADs pass 500-map stress load test with zero segfaults.

---

## Phase 4: H6–H8 — Launch Long Training Run #1 & Automated LOQ Judge
* **Vedesh (OMEN):**
  1. Re-run `train_bc.py` on full demo set with full `HybridDoomNet` (Grid + Slots active), then launch **Long Run #1** (`BC-init → Recurrent PPO + RND + Aux + PLR + Scaffolding`) inside `tmux` on OMEN.
  2. Setup checkpoint export every `250k` env steps to `runs/run1/checkpoints/`.
* **Aryan (LOQ):**
  1. Start automated watch loop (`rsync` new checkpoints from OMEN every 20–30 min → run `score_checkpoint.py` on held-out suite → post `eval/*` to W&B → if new best & passes `validate.py --docker`, copy to staging).
* **Aman (M5):**
  1. Curate additional held-out Freedoom 2 maps (`MAP01–MAP05`) and permissive community single-level WADs (`< 2048` sectors) to test out-of-distribution generalization.
  2. Inspect Run #1 W&B reward breakdowns for reward-hacking (e.g., spinning or farming).
* **Swapnil (M3):**
  1. Record an extra 30 mins of tricky Tier 3–4 backtracking demos (in case Run #2 needs stronger Tier 3 BC priors).
  2. Draft `README.md` skeleton and export clean Mermaid vector architecture diagrams.
* **H8 Gate:** Run #1 stable past `2M+` steps on OMEN with zero memory leaks; LOQ automated judge has scored ≥ 3 checkpoints onto W&B.

---

## Phase 5: H8–H12 — Overnight / Mid-Event Training, Ablations & Tuning
* **Vedesh (OMEN):**
  1. Keep OMEN running at max sustainable worker throughput; inspect W&B every 1–2 hours (or via `/schedule` monitor).
  2. If entropy drops too fast or Tier 3 stalls, adjust `ent_coef`, `rnd_coef`, or `bc_kl` decay and launch **Run #2** from best checkpoint.
* **Aryan (LOQ):**
  1. Between checkpoint evaluations (~5 min every 30 min), run a controlled **Ablation Training Run** on LOQ (`use_grid=False` or `use_rgb=False` on medium batch) so we have genuine quantitative ablation rows for the 2-Page Brief and confirm Grid+Slots wins!
* **Aman (M5):**
  1. Tune PLR tier sampling weights and add procedural geometry variations (L-shaped hallways, pillars, height steps ≤ 24 units, variable lighting) so RGB + rays generalize broadly.
* **Swapnil (M3):**
  1. Write comprehensive `pytest` suites (`tests/test_preprocessing.py`, `tests/test_odometry.py`, `tests/test_mapgen.py`, `tests/test_scorer.py`).
  2. Draft Page 1 of the **2-Page Project Brief PDF** (USP, Hybrid + Scan-Match Grid architecture, Rule compliance).
* **H12 Gate:** Held-out Tier 1 survival `> 80%`, Tier 2 sector coverage `> 65%`, and first Tier 3 keycard pickups/unlocks observed in replays!

---

## Phase 6: H12–H16 — Full Architecture Push & Feature Freeze
* **Vedesh (OMEN):**
  1. Launch/continue **Run #3 (Flagship Full Config)** incorporating any reward/PLR tweaks discovered in Phase 5, with RGB modality dropout (`p=0.25`) active.
* **Aryan (LOQ):**
  1. Profile exact single-thread CPU & GPU memory inside `validate.py --docker` (`torch.set_num_threads(1)`). If mean latency `> 20 ms`, apply `torch.jit` or trim conv channels now before H16 freeze.
* **Aman (M5):**
  1. Watch replay recordings of held-out Tier 3 & Tier 4 failures; adjust `rewards.py` / PLR weights one final time before H16 freeze.
* **Swapnil (M3):**
  1. Build replay rendering script (`record_path` / MP4 side-by-side of RGB + Egocentric Occupancy Grid live overlay — huge visual wow-factor for Stage 2 Desk Review!).
  2. Draft Page 2 of Brief PDF (5-stage training pipeline, ablation table skeleton, honest limitations).
* **H16 Gate (HARD FEATURE FREEZE):** All code changes to `model.py`, `preprocessing.py`, `odometry_grid.py`, and `agent.py` **freeze at H16**. Only weight checkpoints, training configs, tests, and documentation change after H16.

---

## Phase 7: H16–H20 — Mandatory Vanilla-Only Fine-Tune & Checkpoint Lock
* **Vedesh (OMEN):**
  1. Take the highest-scoring checkpoint on held-out eval at H17–H18 and run the **Final Vanilla-Only Fine-Tune** (`scaffolding_alpha = 0.0`, 100% vanilla doors/ammo/health, `SKILL = 3`, `bc_kl = 0.0`, lower `lr = 8e-5`) for `2–4 hours`.
* **Aryan (LOQ):**
  1. Run an expanded 40-map held-out tournament across top 5 candidate checkpoints (pre-finetune vs post-vanilla-finetune) + 4,200-tick endurance test.
  2. Promote the winning checkpoint to `submission/weights/best_agent.pt` via **Git LFS** and verify `python validate.py --docker`.
* **Aman (M5):**
  1. Document `training/README.md` (exact commands to generate maps, record/train BC, run PPO+PLR, and run held-out scorer).
* **Swapnil (M3):**
  1. Populate final quantitative results & ablation numbers from Aryan's tournament into Page 2 of the **2-Page Project Brief PDF**.
  2. Prepare the **10-Question Desk-Review Q&A Cheat Sheet** and 5-minute live demo flow.
* **H20 Gate (CHECKPOINT LOCK):** `submission/weights/best_agent.pt` committed with Git LFS on `main`, passing `validate.py --docker` with `< 25 ms/step`. *(OMEN may keep training a safety candidate until H22 only if strictly drop-in identical architecture.)*

---

## Phase 8: H20–H24 — Cold-Clone Verification, Brief Export & Final Submission
* **Vedesh (OMEN):**
  1. Perform a **fresh `git clone` into an empty `/tmp/clean-verify` folder** (testing LFS pull!), run `docker build -t devforge-doom-final .` and `python validate.py --docker --image devforge-doom-final`.
  2. Verify `git status` clean, `git lfs ls-files` lists weights, total repo size `< 1 GB`, zero secrets or `.env` files in commit history, repository set to **Public**.
* **Aryan (LOQ):**
  1. Audit every line under `submission/` against `RULES.md` (zero `atan2`, zero `A*`/BFS, zero rule trees, empty or strictly valid `submission/requirements.txt`).
  2. Run independent cold-clone `validate.py --docker` check on LOQ.
* **Aman (M5):**
  1. Verify `uv run pytest` passes 100% across all tests and `training/` scripts have clean `--help` / docstrings for manual code reviewers.
* **Swapnil (M3):**
  1. Export final **2-Page Project Brief PDF** (verify strictly `≤ 2 pages`, crisp fonts, embedded vector diagram & metrics table).
  2. Export 60-second showcase MP4 replay (Agent RGB + Live Occupancy Grid) ready on laptop for Stage 2 Desk Review.
  3. Lead a 10-minute team dry-run of the live demo + Q&A.
* **H23.5 Gate (HARD SUBMISSION CUTOFF):** Final commit SHA recorded, GitHub URL + 2-Page PDF submitted on portal 30 minutes ahead of deadline. **Do not push after H23.5.**

---

## Final Pre-Submission Checklist
- [ ] Repository is **Public** and created from `Reinforce-SST/devforge-doom-template` after H0.
- [ ] `git lfs ls-files` shows `submission/weights/best_agent.pt` (repo size `< 1 GB`).
- [ ] `submission/requirements.txt` has only comment lines (or pinned `pkg==ver` tested in Docker).
- [ ] Fresh shallow clone passes `python validate.py --docker` with zero warnings and `< 25 ms/step` latency.
- [ ] `submission/` contains zero hardcoded rule trees, pathfinders, or `atan2` aimbots.
- [ ] `training/` contains complete, documented BC, MapGen, PLR, RND, and Recurrent PPO source code.
- [ ] 2-Page Project Brief PDF is `≤ 2` pages, proofread, and uploaded alongside the repo URL.
