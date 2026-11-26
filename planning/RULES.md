# Track 1 ("Can It Play DOOM?") — Official & Team Rulebook

> **Zero-Tolerance Pre-Start Rule:** Not a single line of repository or project code may be written before the official hackathon clock starts (**H0**). Until H0, only machine setup, reading documentation, caching Docker layers, and practicing keyboard controls are allowed.

---

## 1. Official Hackathon Rules (Condensed from Template `docs/`)

### 1.1 Trained Neural Parameters Only (`docs/RULES.md §1`)
* **Every action decision at evaluation MUST originate from trained neural network weights.**
* **Strictly Banned inside `submission/`:**
  * Hand-written `if/else` rule trees or finite-state machines that select actions.
  * Classical pathfinding or graph search (`A*`, `Dijkstra`, `BFS`, wall-following scripts, frontier planners).
  * Trigonometric aimbots (e.g., computing `atan2` on `visible_entities` angles to turn/shoot directly).
  * Unmodified, off-the-shelf `Stable-Baselines3` (or equivalent plug-and-play baselines with default hyperparameters).
* **Explicitly Allowed (`docs/RULES.md §1`):**
  * Observation preprocessing and feature encoding (e.g., normalizing vitals, circular ray formatting, building an observation-derived egocentric occupancy grid via scan-matching).
  * **Action masking derived strictly from observation fields** (e.g., masking `USE` for a few steps after button-down to prevent door reversal, or masking `ATTACK` when selected non-melee weapon ammo is `0`).
  * Frame stacking and learned recurrent / slot memory modules.
  * Any training-time scaffolding (auto-opening doors, ACS scripts, infinite ammo, dense `privileged_state()` rewards) — **during training only**.

### 1.2 Training vs. Evaluation Isolation (`docs/CONTRACT.md §1`, `docs/RULES.md §2`)
* **Evaluation runs strictly on the unmodified vanilla ViZDoom `1.3.3` engine**, `freedoom2.wad`, `SKILL = 3` ("Hurt me plenty"), `FRAME_SKIP = 4`, and **unseen maps** across Tiers 1–4.
* The scorer **ignores `env/` in your repository** and mounts your repo **read-only** with **no internet access** (only `/tmp` is writable, up to 2 GB).
* `DoomAgent` receives **only** the contract observation dictionary (`rgb`, `depth`, `vitals`, `rays`, `visible_entities`). It never receives `vizdoom.DoomGame` or `privileged_state()`.
* Attempting to inspect the host, read evaluation WAD files, access the network, or exhaust shared GPU/CPU/disk resources results in **immediate disqualification**.

### 1.3 Hardware, Latency & Submission Constraints (`docs/EVALUATION.md`, `docs/SUBMISSION.md`)
* **Design Target (Stricter Envelope):** `1` CPU thread (`torch.set_num_threads(1)`), `≤ 2 GB` VRAM (`≤ 1.5 GB` safe target), `≤ 8 GB` RAM. *(Even if draft `6 GB VRAM / 4 cores` applies, designing for the stricter spec guarantees zero OOMs or timeouts.)*
* **Hard Timeouts (`env/doom_env/limits.py`):**
  * `DoomAgent.__init__`: `≤ 60 s`
  * `DoomAgent.reset`: `≤ 10 s`
  * First `act()` on each map: `≤ 10 s`
  * Subsequent `act()` calls: `≤ 5 s`
  * Average compute budget per map: **`40 ms × decision_steps`** (our internal promotion target: **`< 25 ms/step`** mean on 1 CPU thread).
  * *Penalty on failure:* If `act()` crashes, times out, or returns anything other than shape `(9,)` values in `{0, 1}`, the player stands still for the remainder of that map.
* **Repository & Weights:**
  * Public GitHub repository created **after H0** using **"Use this template"** on `Reinforce-SST/devforge-doom-template`.
  * Total repo size (including Git LFS objects) **`≤ 1 GB`**.
  * Model weights under `submission/weights/` tracked via **Git LFS**.
  * Extra packages in `submission/requirements.txt` must use strict `pkg==version` lines (`≤ 50` lines, no URLs/flags). **Team policy: keep `submission/requirements.txt` empty** by relying solely on packages pre-installed in the official `Dockerfile`.
  * `training/` must contain our real training, mapgen, and curriculum code — **organizers manually audit the code and replays of the top 4–5 teams**.
  * `python validate.py --docker` must pass 100% cleanly on the final pushed commit before deadline, accompanied by the **2-Page Project Brief PDF**.

---

## 2. Team Engineering Rules

| Rule | Policy | Owner / Enforcer |
| :--- | :--- | :--- |
| **Branch Naming** | All work happens on `feat/<owner>-<module>` (e.g., `feat/vedesh-ppo`, `feat/aryan-model`, `feat/aman-mapgen`, `feat/swapnil-bc-docs`). | Everyone |
| **Protected `main`** | **Zero direct pushes to `main`.** Open a PR; merge only after `uv run pytest` and `uv run python validate.py` pass. | Vedesh (OMEN) |
| **Interface Lock (H0.5)** | `Schema.md` tensor shapes, dictionary keys, and function signatures freeze at **H0.5**. Any schema change requires agreement from both producer and consumer before editing. | Vedesh + Aryan |
| **Checkpoint Hygiene** | **Never commit intermediate `.pt` checkpoints to Git.** Sync intermediate checkpoints between OMEN and LOQ via Tailscale (`rsync`/`scp` into `runs/` which is `.gitignore`d). Only Aryan's promoted winner goes into `submission/weights/best_agent.pt` via Git LFS. | Aryan (LOQ) |
| **Zero Secrets in Git** | Never hardcode or commit W&B API keys, Tailscale keys, SSH keys, or `.env` files. Pass `WANDB_API_KEY` via shell environment only. | Everyone |
| **2-Hour Sync Pulse** | Every 2 hours on the even hour (`H2, H4, ..., H22`), post a 3-bullet update in team chat: *(1) Done, (2) Next 2h, (3) Blockers / FPS / Score*. | Everyone |
| **30-Minute Escalation** | **If stuck on any bug, segfault, or setup issue for > 30 minutes, escalate immediately** in team chat and jump on a 5-minute screen share. Never burn 2 hours silently. | Everyone |
| **Time Gates** | **H16 = Feature Freeze** (no new neural layers or mapgen rewrites). **H20 = Checkpoint Lock** (final vanilla fine-tune promoted). **H23.5 = Hard Push Cutoff** (30-min buffer before official deadline). | Vedesh (OMEN) |

---

## 3. AI-Assistant Rules (Antigravity Usage)

1. **No Pre-H0 Code Generation:** Do not prompt Antigravity to generate Python modules, neural networks, or WAD scripts until the hackathon officially begins at H0.
2. **One Workspace, One Branch, One Owner:** Each teammate points Antigravity at their own local clone checked out to their own `feat/<name>-...` branch. Never have two teammates' AI sessions edit the same file simultaneously.
3. **Desk-Review Ownership Rule:** During Stage 2 Desk Review, judges ask live technical questions to verify understanding. **You may not merge AI-generated code unless you can explain how every function in your PR works** (e.g., GAE/BPTT masking for Vedesh, scan-matching & attention masking for Aryan, UDMF sectors & PLR for Aman, BC loss & metrics for Swapnil).
4. **No Runtime AI / LLMs:** `submission/agent.py` is a pure self-contained PyTorch policy. Never import LLM SDKs, API clients, or heavy external frameworks.
5. **Mandatory Rule Audit Prompt before Merge:** Before merging any PR touching `submission/`, run a quick review check verifying zero `atan2`, zero `A*`/BFS, and zero action-selecting `if/else` branches exist in `submission/`.
