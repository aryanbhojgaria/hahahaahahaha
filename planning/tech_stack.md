# Technology Stack & Architectural Choices

## 1. Selected Stack

| Layer | Technology | Why We Chose It | Primary Owner |
| :--- | :--- | :--- | :--- |
| **Simulation & Wrapper** | `vizdoom==1.3.3` + Official `doom_env` (`freedoom2.wad`, Skill 3, `FRAME_SKIP=4`) | Exact evaluation contract (`docs/CONTRACT.md`); runs thousands of spatial raycast ticks/sec on CPU cores. | Vedesh / All |
| **Deep Learning Core** | **PyTorch** (`torch`, CUDA 12 in training, 1-thread CPU / CUDA in eval) + `einops` | Pre-installed in official `Dockerfile`; clean dynamic graphs for variable-length entity masks, scan-matching grid warps (`grid_sample`), and custom BPTT. | Aryan & Vedesh |
| **Sensory Encoders** | Circular 1D Conv (64 rays) + Masked Multi-Head Attention (≤32 entities) + MLP (17 vitals) + Depthwise CNN (`64×48` RGB+Depth w/ modality dropout) | Exploits 360° exact geometry immediately while keeping visual projectile/texture awareness without pixel-RL sample inefficiency. | Aryan |
| **Spatial & Long-Horizon Memory** | **360° Ray Scan-Matching Odometry** + `32×32` 6-ch Egocentric Occupancy Grid + `GRU(256)` + 4 Explicit Objective Slots | Gives true room-level memory and locked-door/keycard bearing retention without needing forbidden global `(x, y, angle)` coordinates or classical `A*`. | Aryan |
| **RL Algorithm** | **Custom Recurrent PPO** (GAE, Clipped Surrogate, Truncated BPTT `L=32`, 9 Bernoulli Heads, Obs Action Masking, Decaying BC-KL) | Avoids SB3 penalties, supports recurrent hidden states + custom action masks natively, and stays rock-solid over 12+ hour runs. | Vedesh |
| **Intrinsic Exploration** | **Random Network Distillation (RND)** on spatial-ray embeddings + Egocentric Grid Cell-Visit Novelty | Drives the agent out of spawn rooms and dead ends even before extrinsic exit/keycard rewards trigger. | Vedesh |
| **Warm-Start Imitation** | **Interactive `pygame` Demo Recorder** + Supervised Behavior Cloning (BC) pretrainer | 1–2 hours of human play teaches strafe-aiming and single-press `USE` door opening in 10 minutes instead of 5M random RL steps. | Swapnil & Vedesh |
| **Procedural Training Maps** | **Custom Parameterized PWAD/UDMF Generator** (`omgifol` + `zdbsp`) + ViZDoom Scenarios + `freedoom2` + Community WADs | Prevents layout memorization; lets us dial room counts, door locks, and ambushes across Tiers 1–4 dynamically. | Aman |
| **Curriculum Engine** | **Prioritized Level Replay (PLR)** + Annealed Engine Scaffolding → Mandatory Vanilla Fine-Tune | Concentrates compute on frontier difficulty seeds while guaranteeing final weights run cleanly on vanilla DOOM rules. | Aman & Vedesh |
| **Checkpoint Selection** | **LOQ Automated Held-Out Scorer** mirroring `EVALUATION.md` + `validate.py --docker` gate | Selects weights by zero-shot generalization and verified `< 25 ms` latency rather than noisy training return. | Aryan |
| **Experiment Tracking** | **Weights & Biases (`wandb`)** + local TensorBoard backup | Shared real-time dashboard across all 4 laptops for FPS, losses, and tier-by-tier held-out scores. | Vedesh & Aryan |
| **Infra, Sync & Storage** | **WSL2 Ubuntu 24.04**, **Tailscale** (`rsync`/SSH), **GitHub** (protected `main`), **Git LFS** | Linux `fork`/`spawn` stability for 16+ ViZDoom workers on Windows laptops; instant cross-OS file sync with Macs; `< 1 GB` LFS compliance. | Vedesh |
| **Testing & QA** | **`pytest`** + Official `validate.py --docker` | Fast regression checks on shapes, odometry, reset isolation, and container limits before every PR merge. | Swapnil & All |
| **Brief & Diagrams** | **Markdown + Mermaid** → 2-Page PDF (Typst / Google Docs / Pandoc) + MP4 replay captures | Crisp vector diagrams and real ablation tables tailored for Stage 2 Desk Review and Finals. | Swapnil |
| **AI Coding Workflow** | **Antigravity IDE** — **Gemini Pro (High)** / **Claude Opus** (Vedesh paid + targeted spikes) + **Gemini Flash** (fast implementation/tests/docs) | Maximizes team velocity within quota limits while keeping human owners in full control of architecture. | All |

---

## 2. Rejected Alternatives (And Why)

| Rejected Option | Reason Rejected |
| :--- | :--- |
| **`Stable-Baselines3` (PPO / RecurrentPPO)** | Explicitly called out as low-scoring / penalized in the problem brief and `docs/RULES.md §1`; awkward dict+mask+custom-memory customization. |
| **Sample Factory (Async APPO)** | Fast ViZDoom framework, but heavy external dependency not in base `Dockerfile` (bloats `requirements.txt` and obscures bespoke architecture during code review). |
| **Custom Async IMPALA / V-Trace** | Higher theoretical throughput, but off-policy recurrent V-trace + shared-memory race conditions routinely burn 6+ hours of debugging in a 24h hackathon. |
| **Pure `120×160` RGB Pixel CNN** | Wastes ~90% of compute learning basic wall/edge filters, runs 5–8× slower on laptop GPUs, and overfits wall textures on unseen Freedoom WADs. |
| **Cloud GPU/CPU Instances (AWS / Azure)** | Local OMEN i9 HX + RTX 5060 plus LOQ RTX 4050 provide plenty of spatial+tiny-RGB throughput with zero network lag, quota delays, or S3 sync friction (Azure kept strictly as disaster backup). |
| **Classical `A*` / Frontier Planner / `atan2` Aiming** | Explicitly prohibited by `docs/RULES.md §1` — instant disqualification during top-5 manual code review. |
