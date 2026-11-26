# Submission Guide

You submit the URL of a **public GitHub repository** created from this template.

## Checklist

- [ ] The repository is **public**, and the URL was sent to organizers before the deadline.
- [ ] `submission/agent.py` defines `DoomAgent` with `__init__`, `reset`, `act` ([CONTRACT.md](CONTRACT.md#6-agent-interface)).
- [ ] Weights are committed under `submission/` (e.g. `submission/weights/`) with **Git LFS**.
- [ ] Total size, including LFS objects, is **≤ 1 GB**.
- [ ] Extra Python packages (only ones not already in the base image) are listed in `submission/requirements.txt` with pinned versions.
- [ ] `python validate.py` passes on your final commit. We recommend `python validate.py --docker`.
- [ ] Training code is in the repo (e.g. `training/`).
- [ ] Your agent does not need network access or write access to the repository at runtime.

## Git LFS

```bash
git lfs install
git lfs track "*.pt" "*.pth" "*.safetensors" "*.ckpt" "*.npz" "*.msgpack" "*.onnx"   # already in .gitattributes
git add .gitattributes submission/weights/model.pt
git commit -m "Add weights"
git push
```

Check that LFS picked up the weights: `git lfs ls-files` should list them.

> **Size tip:** the scorer fetches only the evaluated commit (no history) and only the LFS objects
> it references, so old weight versions don't count. Weights committed
> **without** LFS still count, and they also bloat your history. Use LFS from the start.

## Base image and `requirements.txt`

The official image already includes Python 3.13, `vizdoom`, `gymnasium`, `torch`, `jax`, `flax`,
`numpy`, `scipy`, `opencv-python-headless` and `doom_env`. Exact versions of everything are in
[`docker/requirements.lock`](../docker/requirements.lock).

If you need anything else, list it in `submission/requirements.txt`, one `package==version` per line
(no `-r`/`-e`/`--index-url`/`--find-links` options, URLs or paths; at most 50 packages). These
packages are installed **before** evaluation (with network, under a time limit), and evaluation then
runs **without** network. If the file is invalid or the install fails, the agent can't start: you
get 0 on every map.

## Local validator

```bash
python validate.py                 # quick check in your local environment
python validate.py --docker        # same checks inside the official image (recommended)
```

It checks:

1. Repository size (≤ 1 GB, including LFS objects), and that nothing under `submission/` is
   ignored by `.gitignore`, untracked or uncommitted. The scorer evaluates your **last pushed
   commit**, while the validator runs your working tree, so these would differ.
2. `DoomAgent` imports cleanly and can be called as `DoomAgent(observation_space, action_space, device)`,
   `reset()`, `act(observation)`. `requirements.txt` follows the rules above.
3. A full episode on the test map (or your own with `--map file.wad:MAP01`, repeatable) runs without crashing and returns valid actions.
4. Init time, per-step latency, and per-map agent compute time are within the [timeouts](EVALUATION.md#timeouts).
5. Peak GPU memory is reported (stay under the per-agent cap).
