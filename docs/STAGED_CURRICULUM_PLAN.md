# Staged Curriculum Restart — who trains what

**Why we restarted:** the old run sampled `tier_weights [1, 1.5, 2.5, 3.0]` → ~75% Tier‑3/4 maps and PLR (temperature 0.1) kept
replaying the single hardest level. After 800k steps: `exit=0%`, `ret≈-5.7`. The agent never saw an easy win, so it had nothing to learn from.

**New ladder** (`training/curriculum_stages.py`, config `training/configs/ppo_staged.yaml`, no LLM / no API needed):

| Stage | `--stage-lock` | Maps | Advances when rolling exit-rate ≥ |
|---|---|---|---|
| 0 single_map | 0 | ONE fixed map, no enemies | 80% |
| 1 easy | 1 | Tier 1, random seeds, ~1 enemy/room | 60% |
| 2 medium | 2 | Tier 1–2, doors, ~1.5 enemies/room | 45% |
| 3 hard | 3 | Tier 2/3/4 (keycards, multi-lock) | 30% |
| 4 random | 4 | all tiers mixed | — (final) |

Training seeds stay in `[100000, 999999]` (held-out `[1000, 9999]` untouched). `best_out` is empty, so nothing overwrites `submission/weights/`.

## Division of work (all start from the same warm init = `submission/weights/best_agent.pt`, 806.9k steps)

| Person | Machine | Trains | Command |
|---|---|---|---|
| **Vedesh** | RTX GPU | full chain stage 0 → 1 → 2 → … (auto-advance) + final random fine-tune + merge/eval | `python -u -m training.ppo --config training/configs/ppo_staged.yaml --resume submission/weights/best_agent.pt --device cuda --output-dir runs/staged_main` |
| **Aryan** | Mac | **easy → medium** specialist (stage 1) | `... --stage-lock 1 --num-envs 8 --device cpu --output-dir runs/stage1_aryan` |
| **Aman** | Mac | **medium → hard** specialist (stage 2) | `... --stage-lock 2 --num-envs 8 --device cpu --output-dir runs/stage2_aman` |
| **Swapnil** | Mac | **hard / further** specialist (stage 3) | `... --stage-lock 3 --num-envs 8 --device cpu --output-dir runs/stage3_swapnil` |

Each teammate uses the same flags as the Vedesh command (`--config training/configs/ppo_staged.yaml --resume submission/weights/best_agent.pt`) plus the lock/output flags above.
Everyone: commit `runs/<name>/metrics.csv` (small) and send the best `checkpoints/latest.pt` to Vedesh (LFS or drive). **Do not push to `submission/weights/`.**

## Combine
1. Vedesh scores every checkpoint on the 27-map held-out suite (`score_checkpoint.py`).
2. Try (a) sequential hand-off: best stage‑1 → stage‑2 → stage‑3 → stage‑4 fine-tune, and (b) weight average ("soup") of the specialists + a short stage‑4 random fine-tune.
3. Promote only if held-out total > 504.13 (current submission) and `validate.py` passes. Otherwise keep the current weights.

Caveat: averaging RL policies can hurt (GRU state); that is why step 3 is gated on the held-out score.
