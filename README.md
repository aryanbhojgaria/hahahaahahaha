# Can It Play DOOM? — DevForge × Scaler Hackathon 2026 (Track 1: Deep RL)

> **Zero-shot DOOM agent that self-evolves on procedurally generated maps — no hardcoded world coordinates, no memorized button sequences.**

This repository contains our Track 1 Deep RL submission for **DevForge × Scaler Hackathon 2026**. Our agent learns to play unseen DOOM levels zero-shot: navigating multi-room layouts, finding keycards, unlocking doors, surviving combat, and reaching the exit.

---

## Key Highlights

- **Ego-centric perception** — 64-ray depth/hit-type raycasts, entity vitals, yaw-delta odometry, and a coordinate-free 32×32 local occupancy grid (world XYZ never enters the network).
- **HybridDoomNet (Actor-Critic)** — circular 1-D Conv ray encoder, vitals encoder, 256-d GRU temporal core, 4 × 8-d Objective Memory Slots, an **8-sector Value Compass** auxiliary head, and a 9-action MultiBinary policy head with legal action masking.
- **Self-evolving training loop** — Recurrent PPO + GAE + BPTT with RND curiosity, grid novelty bonus, optional BC-KL warmstart, and **Prioritized Level Replay (PLR)** over a 4-tier procedural UDMF map generator.
- **LLM Curriculum Coach** — queries CreoAI (`@cf/openai/gpt-oss-120b`) to adapt difficulty knobs online from live training metrics, with deterministic offline fallback when `CREO_API_KEY` is unset.
- **Strict evaluation envelope** — ≤ 2 GB VRAM, ≤ 8 GB RAM, 1 CPU thread (`torch.set_num_threads(1)`), and ≤ 40 ms/step verified via `validate.py`.

---

## Installation

### Recommended (`uv` + Python 3.13)

This repository pins exact dependencies in `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

### Standard `pip` (if not using `uv`)

```bash
pip install -e env/
pip install -r submission/requirements.txt
# Or if using a root requirements.txt:
pip install -r requirements.txt
```

> **Platform support:** Linux (x86-64/ARM), macOS on Apple Silicon (macOS 15+), and Windows (x86-64) work natively with `uv sync` and `validate.py`. For the closest match to the Linux evaluation sandbox, use Docker:
>
> ```bash
> docker build -t devforge-doom .
> python validate.py --docker --image devforge-doom
> ```

---

## Usage

### Train

Run the self-evolving PPO curriculum trainer:

```bash
# Using uv
uv run python -m training.ppo --config training/configs/ppo_curriculum.yaml

# Or inside an activated virtual environment
python -m training.ppo --config training/configs/ppo_curriculum.yaml
```

To run offline without LLM Coach API calls, pass an empty `CREO_API_KEY`:

```bash
CREO_API_KEY="" uv run python -m training.ppo --config training/configs/ppo_curriculum.yaml
```

### Validate

Verify the submission contract, single-thread latency budget (≤ 40 ms/step), and memory limits:

```bash
uv run python validate.py
# Or:
python validate.py
```

### Verify Submission Entrypoint

Check that the evaluation agent imports cleanly:

```bash
uv run python -c "from submission.agent import DoomAgent; print('OK')"
# Or:
python -c "from submission.agent import DoomAgent; print('OK')"
```

### Run Tests

```bash
uv run pytest
```

---

## Module Layout

```text
env/                      # doom_env — official ViZDoom Gymnasium wrapper (spaces, sensors, limits)
training/                 # Deep RL training pipeline
  ├── configs/            # YAML configs (ppo_curriculum.yaml, bc.yaml, eval.yaml)
  ├── mapgen/             # 4-tier procedural UDMF map generator & verifier
  ├── bc/                 # Optional behavioural cloning warmstart
  ├── scorer/             # Heldout evaluation suite, checkpoint scorer, replay renderer
  ├── model.py            # HybridDoomNet (RayCircularEncoder, GRU, ObjectiveSlots, Value Compass)
  ├── preprocessing.py    # Ego-centric preprocessing, ActionMask, ModelInputs
  ├── odometry_grid.py    # Coordinate-free scan-matched occupancy grid
  ├── rewards.py          # Coordinate-free sector-based reward shaper
  ├── curriculum_plr.py   # Prioritized Level Replay (PLR) buffer
  ├── envs.py             # VecDoomEnv multiprocessing wrapper
  ├── rnd.py              # Random Network Distillation curiosity + grid novelty
  ├── llm_coach.py        # CreoAI curriculum coach with deterministic offline fallback
  └── ppo.py              # Recurrent PPO + GAE + BPTT training entrypoint
submission/
  ├── agent.py            # DoomAgent evaluation entrypoint (act / reset, 1-thread guard)
  ├── requirements.txt    # Extra pinned pip packages for the evaluation container
  └── weights/            # Trained weights tracked via Git LFS (best_agent.pt)
tests/                    # Unit, integration, and E2E test suite
maps/                     # Test map (two_rooms.wad) + 32 heldout evaluation WADs (maps/heldout/)
docs/                     # Official contract, rules, submission guide, evaluation spec
planning/                 # Architecture design, frozen schemas, PRD, and team runbooks
validate.py               # Pre-submission validator (contract, latency, RAM/VRAM)
PROJECT_BRIEF.md          # 2-page project brief
```

---

## Documentation & Rules

| Doc | Contents |
| --- | --- |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | 2-page architecture overview, USP, and track compliance summary |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Observation/action spaces and `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Ground rules and disqualifiers (no hardcoded XYZ, no rule trees / A* / aimbots) |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring categories, hardware envelope, timeouts |

---

## Team

- **Vedesh** — RL / PPO (`envs.py`, `ppo.py`, `rnd.py`, `llm_coach.py`, `bc/`)
- **Aryan** — Model / Scorer (`model.py`, `preprocessing.py`, `odometry_grid.py`, `submission/agent.py`, `scorer/`)
- **Aman** — MapGen / Rewards (`mapgen/`, `rewards.py`, `curriculum_plr.py`)
- **Swapnil** — Training / Docs (smoke testing, training runs, `README.md`, documentation)
