# Vedesh (HP OMEN 16 — i9 HX + RTX 5060 8 GB) — Step-by-Step Runbook

## 1. Role & Ownership Summary
* **Role:** Team Lead, Repository Gatekeeper & Primary RL Core Engineer (**Paid Google AI Pro Account**).
* **Files You Own:**
  * Repository setup, branch protection, `.gitignore`, Git LFS config
  * `training/configs/bc.yaml`, `training/configs/ppo_curriculum.yaml`
  * `training/envs.py` (Subprocess vectorized `doom_env` workers + scaffolding hooks + procedural WAD hot-swap)
  * `training/ppo.py` (Custom Recurrent PPO: GAE, truncated BPTT `L=32`, 9 Bernoulli heads, action masking, decaying BC-KL, auxiliary loss integration)
  * `training/rnd.py` (Random Network Distillation + grid cell-visit novelty bonus)
  * `training/bc/train_bc.py` (Supervised Behavior Cloning warm-start trainer)
  * `tests/test_envs_and_ppo.py`
* **Interfaces You Provide & Consume (`Schema.md`):**
  * **Consume from Aryan:** `preprocess(obs, odom_state)` (`training/preprocessing.py`) and `HybridDoomNet.forward(batch, h, return_aux=True)` (`training/model.py`).
  * **Consume from Aman:** `MapGen.sample(tier, knobs, seed)` (`training/mapgen/tiers.py`), `RewardShaper.step(priv_prev, priv_now, action)` (`training/rewards.py`), and `PLRBuffer` (`training/curriculum_plr.py`).
  * **Consume from Swapnil:** Recorded human demo episodes `data/demos/**/*.npz`.
  * **Provide to Aryan:** Periodic `.pt` checkpoints in `runs/<run_name>/checkpoints/` synced to LOQ for held-out evaluation.

---

## 2. Phase 0: Pre-Event Machine Setup (Complete Before H0 — Zero Project Code!)

Run inside **Windows PowerShell (Admin)** then **WSL2 Ubuntu 24.04**:
```bash
# 1. Install WSL2 Ubuntu 24.04 (in Windows PowerShell)
wsl --install -d Ubuntu-24.04

# 2. Inside Ubuntu-24.04: core packages, node builder, SSH, tmux
sudo apt update && sudo apt install -y build-essential cmake git git-lfs curl libgl1 libglib2.0-0 libsdl2-dev zdbsp bsp openssh-server rsync tmux htop
git lfs install
sudo service ssh start

# 3. Check exact logical CPU count & GPU visibility
nproc          # Share core count with team; set num_envs = min(20, nproc - 4)
nvidia-smi     # Confirm RTX 5060 8GB is visible

# 4. Install uv + Python 3.13
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.13

# 5. Pre-cache official Docker image & smoke-test ViZDoom + CUDA (delete before H0!)
git clone https://github.com/Reinforce-SST/devforge-doom-template.git /tmp/setup-check
cd /tmp/setup-check
docker build -t devforge-doom .
uv sync
uv run python validate.py --docker --image devforge-doom
uv run python -c "import torch, vizdoom; print('CUDA:', torch.cuda.is_available(), 'ViZDoom:', vizdoom.__version__)"
cd ~ && rm -rf /tmp/setup-check

# 6. Tailscale + W&B Team Login
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4   # Post this IP in team chat so Aryan, Aman & Swapnil can SSH/rsync
uv tool run wandb login
```
* **AI Tip:** Once step 5 passes, run `/learn` in Antigravity so your workspace remembers your exact WSL2 paths, `uv`, and Docker setup.

---

## 3. Phase-by-Phase Execution Checklist

### Phase 1 (H0–H2): Repo Creation & Parallel Subprocess Envs
* **AI Model & Command:** `Gemini Flash` + `/plan` (use `Gemini Pro (High)` for 10-min `/grill-me` schema check with Aryan at H0).
1. **H0:00:** Click **"Use this template"** on `Reinforce-SST/devforge-doom-template` → create **public** GitHub repo → invite Aryan, Aman, Swapnil → protect `main` (require PR).
2. Add `runs/`, `data/demos/`, `/tmp_wads/` to `.gitignore`; verify `.gitattributes` tracks `*.pt`, `*.safetensors`, `*.onnx` via Git LFS.
3. Checkout `feat/vedesh-envs`. Implement `training/configs/ppo_curriculum.yaml` and `training/envs.py`:
   * Use `multiprocessing.get_context("spawn")` workers wrapping `doom_env.make_env`.
   * Return `(obs, shaped_reward, terminated, truncated, privileged_summary)` per step.
   * Support `reset_with_wad(wad_path, map_name, seed, episode_ticks)` so PLR swaps maps cleanly between episodes.
4. Benchmark `16` parallel workers stepping random actions (`render_buffers=True` vs `False`).
* **Done Criteria (H2 Gate):** `envs.py` achieves `> 1,000` steps/sec across 16 workers with zero deadlocks; Aryan's baseline `agent.py` PR merged and passing `validate.py --docker`.

### Phase 2 (H2–H4): Custom Recurrent PPO + RND Smoke Run
* **AI Model & Command:** **`Gemini Pro (High)` or `Claude Opus`** + **`/boost`** *(highest bug-risk math in the whole repo — spend your Paid Pro quota here!)*.
1. Checkout `feat/vedesh-ppo`. Implement `training/ppo.py`:
   * Rollout buffer storing `(num_envs=16, rollout_steps=128)` transitions + initial `RecurrentState` every `bptt_len=32` steps.
   * **Crucial Recurrent Rule:** Inside truncated BPTT (`T=32`), multiply `gru_h` and `obj_slots` by `(1.0 - done[t-1])` at episode boundaries so memory never bleeds across maps!
   * Masked `Bernoulli(logits=masked_logits)` log-probs summed across the `9` buttons + entropy bonus + GAE (`γ=0.99, λ=0.95`) + clipped value/policy loss + Aryan's `aux_loss`.
2. Implement `training/rnd.py` (fixed random target MLP vs predictor MLP on normalized ray+vitals features; normalize intrinsic rewards by running std).
3. Wire Aman's `rewards.py` and run a 15-minute PPO smoke run on `maps/two_rooms.wad` + Aman's Tier 1–2 maps logging to W&B.
* **Done Criteria (H4 Gate):** Smoke run completes 200k+ steps without NaNs; `rollout/ep_return` and `rollout/kills` visibly climb on W&B; `tests/test_envs_and_ppo.py` passes.

### Phase 3 (H4–H6): Behavior Cloning Warm-Start & Decaying BC-KL
* **AI Model & Command:** `Gemini Flash` for `train_bc.py` (`/plan`); `Gemini Pro (High)` for BC-KL integration.
1. Checkout `feat/vedesh-bc-ppo`. Implement `training/bc/train_bc.py`:
   * Load Swapnil & Aman's `.npz` human demo episodes chunked into `T=32` sequences, run `preprocess()` sequentially per episode to populate `OdomGridState`, and optimize masked `BCEWithLogitsLoss` on the 9 buttons (`pos_weight` boost `2.5` on sparse `USE` and `ATTACK` presses!).
2. Keep a frozen copy of the pretrained `bc_model` in `ppo.py` and add `bc_kl_beta * Bernoulli_KL(pi_theta || pi_bc)` with `bc_kl_beta` decaying linearly `0.20 → 0.0` over `4M` steps.
3. Merge Aryan's `feat/aryan-grid-slots` and Aman's `feat/aman-tier34-plr`, train BC checkpoint (~5–10 mins on RTX 5060), and push `bc_warmstart.pt` to Aryan's LOQ scorer.
* **Done Criteria (H6 Gate):** `bc_warmstart.pt` scores `≥ 2.5×` random baseline on LOQ held-out Tiers 1–2 and opens doors cleanly without bouncing.

### Phase 4 (H6–H8): Launch Long Training Run #1 (`tmux`)
* **AI Model & Command:** `Gemini Flash` + `/schedule` (hourly health check).
1. Launch **Long Run #1** inside `tmux new -s run1`:
   * Full `HybridDoomNet` (Rays + Entities + Vitals + `64×48` RGB w/ `0.25` dropout + `32×32` Scan-Match Grid + Slots), initialized from `bc_warmstart.pt`, PLR over Tiers 1–4, RND + Grid novelty active.
2. Save checkpoints every `250k` steps (`runs/run1/ckpt_*.pt`) so Aryan's LOQ pulls them automatically via Tailscale.
3. Set up `/schedule` every 60 mins to check W&B / stdout logs for worker hangs, memory growth, or entropy collapse.
* **Done Criteria (H8 Gate):** Run #1 past `2M+` steps smoothly; GPU utilization healthy; zero RAM creep.

### Phase 5 & 6 (H8–H16): Mid-Event Iteration, Pro Quota Support & Feature Freeze
* **AI Model & Command:** **`Gemini Pro (High)` / `Claude Opus`** + `/boost` for diagnostics & helping teammates if they hit Free-tier limits.
1. Monitor W&B `eval/tier1..4` curves coming from Aryan's LOQ judge.
2. If Tier 3 keycard pickup lags at H10–H12, sync with Aman to boost PLR Tier 3 sampling weight or keycard reward, and launch **Run #2 / Run #3** warm-started from current best checkpoint.
3. **Act as Team Paid-Pro Backup:** If Aryan or Aman hits a Free-tier reasoning quota wall before H16, pull their branch on a secondary terminal and run `/boost` for them.
4. **Enforce H16 Feature Freeze:** Merge last model/mapgen/preprocessing PRs by **H16:00**.
* **Done Criteria (H16 Gate):** Architecture frozen; strong upward held-out curves across Tiers 1–3.

### Phase 7 (H16–H20): Mandatory Vanilla-Only Fine-Tune
* **AI Model & Command:** `Gemini Pro (High)` (hyperparameter sanity check).
1. At H17–H18, take the highest-scoring held-out checkpoint and launch `run_vanilla_finetune`:
   * `scaffolding_alpha = 0.0` (100% vanilla DOOM doors, ammo, health, `SKILL=3`), `bc_kl_beta = 0.0`, `lr = 8e-5`, `ent_coef = 0.005`, pure distribution matching unseen evaluation Tiers 1–4.
2. Stream checkpoints every `150k` steps to Aryan's LOQ 40-map tournament.
* **Done Criteria (H20 Gate):** Vanilla fine-tuned checkpoint wins LOQ tournament and is committed to `submission/weights/best_agent.pt` via Git LFS.

### Phase 8 (H20–H24): Clean-Clone Docker Audit & Final Submission
* **AI Model & Command:** `Gemini Flash`.
1. Clean `training/` scripts, verify `uv run pytest` passes 100%, merge final docs/brief PRs.
2. **Mandatory Cold-Clone Test on OMEN:**
   ```bash
   git clone https://github.com/<org>/<repo>.git /tmp/final-audit
   cd /tmp/final-audit
   git lfs ls-files          # Must list submission/weights/best_agent.pt
   du -sh .                  # Must be well under 1 GB
   python validate.py --docker --image devforge-doom
   ```
3. Confirm repo is **Public**, zero secrets/keys in git log, and submit GitHub URL + Swapnil's 2-Page PDF by **H23.5**.

---

## 4. Tests You Must Write (`tests/test_envs_and_ppo.py`)
* `test_bptt_done_masking()`: Verify hidden state `gru_h` and `obj_slots` zero out cleanly at step `t` when `done[t-1] == True` inside a 32-step chunk.
* `test_ppo_overfit_single_room()`: 60-second sanity test proving PPO loss decreases and action probabilities shift toward a dummy positive reward.
* `test_subprocess_env_reload()`: Verify worker swaps between two different WAD paths across `5` resets without leaking file descriptors.

---

## 5. If Things Go Wrong (Vedesh's Fallbacks)
* **PPO forgets BC door/aiming skills in first 500k steps:** Freeze actor + encoders for first `50` updates so Critic `V(s)` calibrates first, raise `bc_kl_beta_init` to `0.4`, lower `lr` to `1e-4`.
* **ViZDoom workers leak RAM or hang after 500 episodes:** Recycle subprocess worker every `100` episodes inside `envs.py` (`worker_max_episodes = 100`) and wrap `step()` in a 15s pipe timeout that restarts the single worker transparently.
* **Hybrid RGB slows OMEN SPS below 500:** Run 12 workers with `render_buffers=False` (leveraging `rgb_dropout` zeroing!) and 4 workers with `render_buffers=True`, doubling throughput while keeping visual branch calibrated.
