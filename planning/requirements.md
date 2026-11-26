# Hardware, Software & Pre-Event Setup Requirements

> **Important:** Complete **100% of Section 3 before H0**. Do **not** create the hackathon GitHub repository or write agent/training code until H0 starts.

---

## 1. Team Hardware & OS Allocation

| Teammate | Machine | Specs | Target OS / Runtime | Primary Role |
| :--- | :--- | :--- | :--- | :--- |
| **Vedesh** | HP OMEN 16 | Intel Core i9 HX (`14900HX`/`14650HX` — verify cores), RTX 5060 8 GB, Windows 11 | **WSL2 Ubuntu 24.04** + Docker Desktop | Primary RL Trainer (`16–20` parallel `doom_env` workers + CUDA PPO updates) & Repo Lead |
| **Aryan** | Lenovo LOQ | Intel Core i5-13th Gen, RTX 4050, Windows 11 | **WSL2 Ubuntu 24.04** + Docker Desktop | Model/Grid Architect, Continuous Held-Out Scorer Judge, Latency Profiler & Ablation Runner |
| **Aman** | MacBook Air M5 | Apple Silicon M5, macOS | **macOS native (`arm64`)** + SSH to OMEN via Tailscale | Procedural UDMF Map Generator, Node Builder, Reward Shaper & PLR Curriculum |
| **Swapnil** | MacBook Air M3 *(joins +3–4h)* | Apple Silicon M3, macOS | **macOS native (`arm64`)** + SSH to OMEN via Tailscale | Human Demo Recorder & Gameplay Demos, Unit Tests, 2-Page Brief PDF, Replays & Q&A Prep |

---

## 2. Software Dependencies Breakdown

### 2.1 Pre-Installed in Official Evaluation Image (`Dockerfile` / `docker/requirements.in`)
These packages are **guaranteed present at evaluation** (Python `3.13` inside Ubuntu `24.04` `/opt/venv`). **`submission/agent.py` must ONLY import from these so `submission/requirements.txt` stays empty:**
* Python `3.13`
* `vizdoom==1.3.3` (includes `freedoom2.wad`)
* `gymnasium>=1.0`
* `numpy>=2.0`
* `scipy`
* `opencv-python-headless` (`cv2` — used for fast `64×48` RGB downsampling)
* `torch` (CUDA 12 wheel in Docker; supports both `"cuda"` and `"cpu"`)
* `einops`
* `jax[cuda12]`, `flax`, `optax` *(unused by us; we standardize on PyTorch)*
* `tqdm`, `tensorboard`
* `doom_env` (official wrapper installed from `/opt/doom_env_src`)

### 2.2 Training & Development Extras (Install in Local `uv` Env Only — Never in `submission/agent.py`)
* **Python packages (`uv add --dev ...`):**
  * `wandb` — live shared training & held-out evaluation dashboard
  * `pytest>=8` — unit and contract test runner
  * `pyyaml` — clean experiment configs (`training/configs/*.yaml`)
  * `omgifol` — lightweight Python WAD lump reader/writer for assembling procedural PWADs
  * `pygame` — cross-platform window + keyboard state capture for `training/bc/record_demos.py` on macOS/WSLg
* **System / CLI utilities:**
  * `uv` (`>=0.9`) — fast Python 3.13 environment & lockfile manager
  * `git` + `git-lfs` — version control and model weight storage
  * `docker` (Docker Desktop with WSL2 backend on OMEN & LOQ) — official container validation
  * `tailscale` — zero-config mesh VPN linking all 4 laptops (`omen`, `loq`, `aman-m5`, `swapnil-m3`)
  * `zdbsp` (or `bsp` via `apt install bsp` / `brew`) — fast BSP node builder so procedural UDMF/Doom WADs load cleanly in ViZDoom

---

## 3. Pre-Event Setup Checklist & Verification Commands

### 3.1 Vedesh (OMEN) & Aryan (LOQ) — Windows + WSL2 + CUDA + Docker
1. **Install WSL2 Ubuntu 24.04 & System Packages:**
   ```bash
   wsl --install -d Ubuntu-24.04
   # Inside Ubuntu-24.04 terminal:
   sudo apt update && sudo apt install -y build-essential cmake git git-lfs curl libgl1 libglib2.0-0 libsdl2-dev zdbsp bsp rsync tmux htop
   git lfs install
   ```
2. **Verify CPU Cores & NVIDIA GPU inside WSL2:**
   ```bash
   nproc                     # Vedesh: note exact logical core count to size num_envs!
   nvidia-smi                # Must show RTX 5060 (OMEN) / RTX 4050 (LOQ)
   ```
3. **Install `uv` & Python 3.13:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.13
   ```
4. **Install Docker Desktop (Windows) & Pre-Build Official Eval Image:**
   * Enable *Settings → Resources → WSL Integration → Ubuntu-24.04*.
   * Clone template to `/tmp/setup-check` (delete before H0!) and cache the Docker build:
   ```bash
   git clone https://github.com/Reinforce-SST/devforge-doom-template.git /tmp/setup-check
   cd /tmp/setup-check
   docker build -t devforge-doom .
   uv sync
   uv run python validate.py --docker --image devforge-doom
   uv run python -c "import torch, vizdoom; print('CUDA:', torch.cuda.is_available(), 'ViZDoom:', vizdoom.__version__)"
   cd ~ && rm -rf /tmp/setup-check
   ```
5. **Enable SSH Server on OMEN WSL2 + Tailscale:**
   ```bash
   sudo apt install -y openssh-server && sudo service ssh start
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   tailscale ip -4           # Share this IP with Aryan, Aman, Swapnil
   ```
6. **Login to Weights & Biases:**
   ```bash
   uv tool run wandb login   # Or export WANDB_API_KEY in ~/.bashrc
   ```

### 3.2 Aman (MacBook M5) & Swapnil (MacBook M3) — macOS Setup
1. **Install Homebrew Utilities, `uv`, Git LFS & Tailscale:**
   ```bash
   brew install cmake sdl2 boost git git-lfs tailscale
   git lfs install
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.13
   ```
   *(Check if `brew install zdbsp` or building `zdbsp` from source works on macOS; alternatively build pure classic Doom binary lumps in Python + `bsp` or test node builds over SSH on OMEN.)*
2. **Smoke-Test `vizdoom==1.3.3` & `pygame` on Apple Silicon (Delete Before H0):**
   ```bash
   git clone https://github.com/Reinforce-SST/devforge-doom-template.git /tmp/setup-check
   cd /tmp/setup-check
   uv sync
   uv run python validate.py
   uv run python -c "import torch, vizdoom; print('MPS:', torch.backends.mps.is_available(), 'ViZDoom:', vizdoom.__version__)"
   cd ~ && rm -rf /tmp/setup-check
   ```
3. **Verify Tailscale SSH into Vedesh's OMEN:**
   ```bash
   tailscale status
   ssh <vedesh-user>@<omen-tailscale-ip> "nvidia-smi"
   ```
4. **Control Practice (Especially Swapnil & Aman — 15 mins before H0):**
   * Practice playing Freedoom 2 with keyboard-only controls (`W/S` forward/back, `A/D` strafe, `Left/Right` turn, `Space` single-tap `USE` on doors without double-tapping, `Ctrl`/`J` attack) so H4–H6 demo recordings are crisp.
