# Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"

Starter template & active team repository for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**, synced from [`Reinforce-SST/devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template) (`63b90c441ca92bcdf043de9ee3930ac6370a44bc`) at **H0** and hosted on `origin/main` (`https://github.com/aryanbhojgaria/hahahaahahaha.git`).

> For decades, *"Can it run DOOM?"* was the ultimate hardware test. Now: **can your neural network
> actually play DOOM?** Train an agent that plays unseen levels **zero-shot**: survive ambushes,
> navigate multi-room layouts, find keycards, unlock doors and reach the exit.

No dataset is provided. Your agent learns by interacting with the engine.

## Quickstart

1. Clone this repository (`https://github.com/aryanbhojgaria/hahahaahahaha.git`).
2. Set up the environment (Python 3.13, [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync                 # installs vizdoom, gymnasium, torch, training deps and env/ in editable mode
   uv run pytest
   uv run python validate.py
   ```

   Or use the official image, which is exactly what the scorer runs:

   ```bash
   docker build -t devforge-doom .            # or: docker pull <official image tag, announced>
   python validate.py --docker --image devforge-doom   # installs requirements.txt, then runs sandboxed like the scorer
   ```

3. Replace the random baseline in `submission/agent.py` with your trained agent.
4. Put weights in `submission/weights/` (Git LFS) and training code in `training/`.
5. Run `python validate.py --docker --image devforge-doom` (and `uv run pytest`) before pushing completed milestones to `origin/main`.

**Platforms:** Linux (x86-64/ARM), macOS on Apple Silicon (macOS 15+) and Windows (x86-64) all
work with `uv sync` and `validate.py`. Intel Macs have no `vizdoom` 1.3.3 wheel: use Docker or a
Linux machine. Evaluation runs on Linux, so `validate.py --docker` is the closest check.

## Repository layout

```
env/                  official ViZDoom Gymnasium wrapper (doom_env): spaces, sensors, protocol, limits
maps/                 one small test map (NOT an evaluation map); bring your own for training
submission/agent.py   your DoomAgent: the evaluation entry point
submission/weights/   model weights (Git LFS)
training/             your training pipeline
validate.py           run before submitting
tests/                tests for the wrapper, validator, and RL pipeline (uv run pytest)
docs/                 contract, rules, submission guide, evaluation details
planning/             frozen schemas, rules, PRD, architecture design, and team runbooks
template_reference/   verbatim snapshot of Reinforce-SST/devforge-doom-template for diffing/reference
```

## Using the environment

```python
from doom_env import make_env

env = make_env("maps/two_rooms.wad")     # test map; use your own maps for training
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Observations are a dict of `rgb`, `depth`, `vitals`, `rays` and `visible_entities`, all
ego-centric and partially observable. See [docs/CONTRACT.md](docs/CONTRACT.md).
The wrapper's `reward` is a basic default for convenience. Shape your own reward for training.

**Compute tip:** training only on raw pixels is expensive. The `rays`, `visible_entities` and
`vitals` signals run thousands of steps per second on a CPU. Hybrid pixel + spatial models are welcome.

## Hardware at evaluation

One NVIDIA **RTX 3090 (24 GB)** shared between evaluation workers. Each agent gets **2 GB VRAM,
8 GB RAM and 1 CPU thread**, so keep models compact and inference fast
(see [hardware](docs/EVALUATION.md#hardware) and [timeouts](docs/EVALUATION.md#timeouts)).
`validate.py` runs your agent under the same CPU/RAM limits and reports GPU memory.

## Documentation

| Doc | What's inside |
| --- | --- |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Observation/action spaces and the `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Ground rules: what's allowed, what gets you disqualified |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring categories, timeouts, replays |

## Rules in one minute

- Actions must come from **trained neural parameters**: no rule trees, no `A*`/Dijkstra, no `atan2` aimbots,
  no unmodified off-the-shelf SB3.
- Change anything you want **during training**. Evaluation uses the **vanilla engine**, the
  official wrapper and **unseen maps**.
- Create the repo during the hackathon. Libraries are fine; the core architecture and training must be yours.

## Team Planning Hub & Architecture Docs

* [`Yugaantar DevForge Tracks.md`](Yugaantar%20DevForge%20Tracks.md) — Official hackathon problem statements & evaluation funnel.
* [`Track1_DOOM_Game_Plan.md`](Track1_DOOM_Game_Plan.md) — Master 24-hour strategy, architecture, timeline, and open items.
* [`planning/`](planning/) — Detailed specifications & team runbooks:
  * [`RULES.md`](planning/RULES.md) — Official disqualifiers, allowed masking/preprocessing, Git/PR hygiene, and AI rules.
  * [`prd.md`](planning/prd.md) — Success metrics, functional/non-functional requirements, and gates.
  * [`requirements.md`](planning/requirements.md) — Hardware/OS allocation, Docker vs local packages, and Phase 0 setup commands.
  * [`tech_stack.md`](planning/tech_stack.md) — Layer-by-layer stack & rejected alternatives.
  * [`Schema.md`](planning/Schema.md) — Frozen tensor shapes, `.npz` demo format, `.pt` checkpoint dict, JSON manifests, W&B keys, and YAML configs.
  * [`Design.md`](planning/Design.md) — Mermaid architecture/sequence diagrams, scan-matching odometry, reward table, and 1-thread latency budget.
  * [`Final_Implementation_Plan.md`](planning/Final_Implementation_Plan.md) — Phase 0 → Phase 8 master schedule, milestone gates, and submission checklist.
  * [`AI_Usage_Guide.md`](planning/AI_Usage_Guide.md) — Paid AI Pro vs Free-tier model allocation and Antigravity slash commands.
* **Individual Runbooks (`planning/team/` — 3-Person Team):**
  * [`Vedesh_OMEN.md`](planning/team/Vedesh_OMEN.md) — RL Core (`envs.py`, self-evolving `ppo.py`, `rnd.py`, `llm_coach.py`, optional `bc/`), main training runs, repo lead & `pytest` suite
  * [`Aryan_LOQ.md`](planning/team/Aryan_LOQ.md) — `preprocessing.py`, `odometry_grid.py`, `model.py` (8-sector ego-directional value compass + objective slots), `submission/agent.py`, random-map `scorer/`, `render_replay.py`
  * [`Aman_M5.md`](planning/team/Aman_M5.md) — Extreme domain-randomized `mapgen/`, `rewards.py`, self-evolving `curriculum_plr.py`, root `README.md` & 2-Page Project Brief PDF
* [`template_reference/`](template_reference/) — Snapshot of `Reinforce-SST/devforge-doom-template` (`docs/`, `env/doom_env/`, `submission/agent.py`, `validate.py`, `Dockerfile`, `tests/`) for offline/team reference.
