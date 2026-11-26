"""Targeted audit & regression tests covering Tasks 1 through 7."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from submission.agent import DoomAgent, load_strict_checkpoint
from training.curriculum_stages import StageCurriculum
from training.model import HybridDoomNet, RecurrentState
from training.ppo import bptt_unroll, compute_gae, masked_bernoulli_log_prob_entropy, save_checkpoint
from training.rewards import RewardShaper, RewardWeights
from training.rnd import RNDModule
from training.scorer.score_checkpoint import score_checkpoint


# ---------------------------------------------------------------------------
# TASK 1: Strict checkpoint loading in DoomAgent and score_checkpoint
# ---------------------------------------------------------------------------
def test_task1_strict_loading_fails_on_missing_lfs_corrupt_and_mismatched(tmp_path: Path) -> None:
    model = HybridDoomNet(use_rgb=True, use_grid=True, rgb_dropout_p=0.0, gru_dim=256, num_slots=4, slot_dim=8)

    # 1. Missing file
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError):
        load_strict_checkpoint(missing, model)
    with pytest.raises(FileNotFoundError):
        DoomAgent(checkpoint_path=missing)
    with pytest.raises(FileNotFoundError):
        score_checkpoint(checkpoint_path=missing)

    # 2. Git LFS pointer file
    lfs_ptr = tmp_path / "lfs_pointer.pt"
    lfs_ptr.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 123456\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        DoomAgent(checkpoint_path=lfs_ptr)

    # 3. Corrupt binary file
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"\x00\xffnot a valid pickle or zip archive")
    with pytest.raises(RuntimeError, match="Corrupt or unreadable"):
        DoomAgent(checkpoint_path=corrupt)

    # 4. Missing / unexpected / shape-mismatched keys
    bad_sd = dict(model.state_dict())
    first_key = next(iter(bad_sd))
    bad_sd.pop(first_key)
    bad_sd["unexpected_extra_weight"] = torch.zeros(3)
    bad_ckpt = tmp_path / "mismatched.pt"
    torch.save({"model_state_dict": bad_sd}, bad_ckpt)
    with pytest.raises(RuntimeError):
        DoomAgent(checkpoint_path=bad_ckpt)

    # 5. Explicit allow_untrained=True opt-in works, while default refuses None
    with pytest.raises(FileNotFoundError):
        DoomAgent(checkpoint_path=None)
    untrained = DoomAgent(checkpoint_path=None, allow_untrained=True)
    assert untrained.model is not None


# ---------------------------------------------------------------------------
# TASK 2: No PPO config writes directly to submission/weights/
# ---------------------------------------------------------------------------
def test_task2_no_training_config_writes_to_submission_weights() -> None:
    cfg_dir = Path("training/configs")
    for yml in sorted(cfg_dir.glob("*.yaml")):
        cfg = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        best_out = str(cfg.get("best_out") or "")
        assert "submission/weights" not in best_out.replace("\\", "/"), (
            f"{yml} must not write directly to submission/weights/ (got best_out={best_out!r})"
        )


# ---------------------------------------------------------------------------
# TASK 3: Full training-state save/reload + identical synthetic update match
# ---------------------------------------------------------------------------
def test_task3_full_resume_state_and_identical_synthetic_update(tmp_path: Path) -> None:
    torch.manual_seed(123)
    cfg = {"d_model": 64, "gru_dim": 64, "num_slots": 2, "slot_dim": 4, "use_rgb": False, "use_grid": False}
    model_a = HybridDoomNet(cfg)
    rnd_a = RNDModule(feature_dim=548)
    opt_a = torch.optim.Adam(list(model_a.parameters()) + list(rnd_a.predictor.parameters()), lr=1e-3)
    stager_a = StageCurriculum(stage_idx=1, min_episodes=10, window=10, seed=11)
    for _ in range(4):
        lv = stager_a.sample()
        stager_a.report_episode(True, level_id=lv["level_id"], score=2.5)

    # Run 1 warmup step so Adam exp_avg / exp_avg_sq and RND RMS are non-trivial
    dummy_in = {
        "rays": torch.randn(2, 4, 8, 64),
        "ent_feats": torch.zeros(2, 4, 32, 5),
        "ent_types": torch.zeros(2, 4, 32, dtype=torch.long),
        "ent_mask": torch.zeros(2, 4, 32, dtype=torch.bool),
        "vitals": torch.randn(2, 4, 36),
        "rgb_depth": torch.zeros(2, 4, 4, 48, 64),
        "grid": torch.zeros(2, 4, 6, 32, 32),
        "action_mask": torch.ones(2, 4, 9, dtype=torch.bool),
        "d_theta": torch.zeros(2, 4),
    }
    dones = torch.zeros(2, 4)
    h0 = RecurrentState.zeros(2, 64, 2, 4)

    logits, vals, _, _ = bptt_unroll(model_a, dummy_in, dones, init_h=h0, return_aux=False, rgb_dropout_p=0.0)
    rnd_feats = torch.cat([dummy_in["rays"].flatten(2), dummy_in["vitals"]], dim=-1).reshape(-1, 548)
    _, rnd_loss = rnd_a.compute_intrinsic_reward(rnd_feats, update_stats=True)
    loss0 = logits.pow(2).mean() + vals.pow(2).mean() + rnd_loss
    opt_a.zero_grad()
    loss0.backward()
    opt_a.step()

    # Save full state
    ckpt_path = tmp_path / "full_resume.pt"
    tstate = {
        "optimizer_state_dict": opt_a.state_dict(),
        "rnd_state_dict": rnd_a.full_state_dict(),
        "stager_state_dict": stager_a.state_dict(),
        "rng_states": {"torch": torch.get_rng_state()},
        "best_val_score": 1234.5,
        "schedule_total_updates": 100,
    }
    save_checkpoint(ckpt_path, model=model_a, model_config=cfg, global_step=2048, update_idx=2, training_state=tstate)

    # Reload into fresh objects
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_b = HybridDoomNet(cfg)
    rnd_b = RNDModule(feature_dim=548)
    opt_b = torch.optim.Adam(list(model_b.parameters()) + list(rnd_b.predictor.parameters()), lr=1e-3)
    stager_b = StageCurriculum(stage_idx=0, min_episodes=10, window=10, seed=99)

    model_b.load_state_dict(loaded["model_state_dict"], strict=True)
    opt_b.load_state_dict(loaded["training_state"]["optimizer_state_dict"])
    rnd_b.load_full_state_dict(loaded["training_state"]["rnd_state_dict"])
    stager_b.load_state_dict(loaded["training_state"]["stager_state_dict"])

    assert stager_b.stage_idx == stager_a.stage_idx == 1
    assert np.allclose(rnd_b.obs_rms.mean, rnd_a.obs_rms.mean)
    assert np.allclose(rnd_b.rew_rms.var, rnd_a.rew_rms.var)

    # Verify one identical synthetic update gives matching weights & loss
    for mdl, rnd_m, opt_m in ((model_a, rnd_a, opt_a), (model_b, rnd_b, opt_b)):
        lg, vl, _, _ = bptt_unroll(mdl, dummy_in, dones, init_h=h0, return_aux=False, rgb_dropout_p=0.0)
        _, rl = rnd_m.compute_intrinsic_reward(rnd_feats, update_stats=False)
        ls = lg.pow(2).mean() + vl.pow(2).mean() + rl
        opt_m.zero_grad()
        ls.backward()
        opt_m.step()

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)
    for pa, pb in zip(rnd_a.predictor.parameters(), rnd_b.predictor.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


# ---------------------------------------------------------------------------
# TASK 4: Damaging non-lethal shot is NOT penalized
# ---------------------------------------------------------------------------
def test_task4_damaging_non_lethal_shot_is_not_penalized() -> None:
    shaper = RewardShaper()
    shaper.reset([])
    p0 = {"x": 0.0, "y": 0.0, "kills": 0, "items": 0, "damage_taken": 0.0, "dead": False, "finished": False}
    p1 = {"x": 0.0, "y": 0.0, "kills": 0, "items": 0, "damage_taken": 0.0, "dead": False, "finished": False}
    act_shoot = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    act_noop = np.zeros(9, dtype=np.int8)

    r_shoot = shaper.step(p0, p1, act_shoot)
    r_noop = shaper.step(p0, p1, act_noop)
    assert shaper.last_components.get("wasted_shot", 0.0) == 0.0
    assert r_shoot == pytest.approx(r_noop, abs=1e-6)


# ---------------------------------------------------------------------------
# TASK 5: GAE hand-computed tests for continuing, terminated, and truncated
# ---------------------------------------------------------------------------
def test_task5_gae_continuing_terminated_and_truncated_hand_computed() -> None:
    gamma = 0.9
    lam = 0.8

    # 3-step rollout, B=3 envs:
    # Env 0: continuing across all 3 steps (next_value = 10.0)
    # Env 1: real terminal at t=1 (terminated=1, truncated=0) -> NO bootstrap at t=1, trace stops
    # Env 2: truncated at t=1 (terminated=0, truncated=1, V(final_obs)=20.0) -> bootstraps 20.0 at t=1, trace stops
    rewards = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
        ],
        dtype=torch.float32,
    )
    dones = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    truncated = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    trunc_values = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    next_val = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)

    adv, ret = compute_gae(
        rewards,
        values,
        dones,
        next_val,
        gamma=gamma,
        gae_lambda=lam,
        truncated=truncated,
        trunc_values=trunc_values,
    )

    # Hand-computed values:
    # Step t=2 (continuing for all 3): delta2 = 3 + 0.9*10 - 6 = 6.0
    d2 = 3.0 + gamma * 10.0 - 6.0
    # Env 0 (continuing):
    d1_e0 = 2.0 + gamma * 6.0 - 5.0
    a1_e0 = d1_e0 + gamma * lam * d2
    d0_e0 = 1.0 + gamma * 5.0 - 4.0
    a0_e0 = d0_e0 + gamma * lam * a1_e0

    # Env 1 (real terminal at t=1): NO bootstrap, NO carry from t=2
    a1_e1 = 2.0 - 5.0
    a0_e1 = (1.0 + gamma * 5.0 - 4.0) + gamma * lam * a1_e1

    # Env 2 (truncated at t=1 with V_final=20.0): bootstraps 20.0, NO carry from t=2
    a1_e2 = 2.0 + gamma * 20.0 - 5.0
    a0_e2 = (1.0 + gamma * 5.0 - 4.0) + gamma * lam * a1_e2

    assert adv[2, 0].item() == pytest.approx(d2, abs=1e-5)
    assert adv[1, 0].item() == pytest.approx(a1_e0, abs=1e-5)
    assert adv[0, 0].item() == pytest.approx(a0_e0, abs=1e-5)

    assert adv[1, 1].item() == pytest.approx(a1_e1, abs=1e-5)
    assert adv[0, 1].item() == pytest.approx(a0_e1, abs=1e-5)

    assert adv[1, 2].item() == pytest.approx(a1_e2, abs=1e-5)
    assert adv[0, 2].item() == pytest.approx(a0_e2, abs=1e-5)
    assert torch.allclose(ret, adv + values, atol=1e-5)


# ---------------------------------------------------------------------------
# TASK 6: Rollout vs PPO update log-probabilities match with rgb_dropout_p=0.0
# ---------------------------------------------------------------------------
def test_task6_rollout_and_update_log_probs_match_without_rgb_dropout() -> None:
    torch.manual_seed(7)
    model = HybridDoomNet(use_rgb=True, use_grid=True, rgb_dropout_p=0.0, gru_dim=64, num_slots=2, slot_dim=4)
    model.eval()

    T, B = 6, 2
    obs_seq = {
        "rays": torch.randn(B, T, 8, 64),
        "ent_feats": torch.randn(B, T, 32, 5),
        "ent_types": torch.zeros(B, T, 32, dtype=torch.long),
        "ent_mask": torch.ones(B, T, 32, dtype=torch.bool),
        "vitals": torch.randn(B, T, 36),
        "rgb_depth": torch.randn(B, T, 4, 48, 64),
        "grid": torch.randn(B, T, 6, 32, 32),
        "action_mask": torch.ones(B, T, 9, dtype=torch.bool),
        "d_theta": torch.zeros(B, T),
    }
    actions = torch.randint(0, 2, (B, T, 9)).float()
    dones = torch.zeros(B, T)
    h0 = RecurrentState.zeros(B, 64, 2, 4)

    # Rollout step-by-step log_probs
    h = h0
    rollout_lps = []
    with torch.no_grad():
        for t in range(T):
            step_in = {k: v[:, t] for k, v in obs_seq.items()}
            logits_t, _, _, h = model(step_in, h, return_aux=False)
            lp_t, _ = masked_bernoulli_log_prob_entropy(logits_t, step_in["action_mask"], actions[:, t])
            rollout_lps.append(lp_t)
    rollout_lp = torch.stack(rollout_lps, dim=1)

    # PPO update bptt_unroll log_probs with model.train() and rgb_dropout_p=0.0
    model.train()
    upd_logits, _, _, _ = bptt_unroll(model, obs_seq, dones, init_h=h0, return_aux=False, rgb_dropout_p=0.0)
    upd_lp, _ = masked_bernoulli_log_prob_entropy(upd_logits, obs_seq["action_mask"], actions)

    assert torch.allclose(rollout_lp, upd_lp, atol=1e-5)


# ---------------------------------------------------------------------------
# TASK 7: Spawn-relative exploration cell grid preserves translation invariance
# ---------------------------------------------------------------------------
def test_task7_spawn_relative_exploration_cells_are_translation_invariant() -> None:
    w = RewardWeights(new_cell=0.06, loop_penalty=-0.03, forward_progress=0.004)
    sa, sb = RewardShaper(w), RewardShaper(w)
    sa.reset([])
    sb.reset([])

    # Shift map by non-multiple of 80 (+37, +19)
    shift_x, shift_y = 37.0, 19.0
    pts = [(10.0, 10.0), (50.0, 10.0), (95.0, 10.0), (140.0, 10.0)]
    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        pa0 = {"x": x0, "y": y0, "kills": 0, "items": 0, "damage_taken": 0.0, "dead": False, "finished": False}
        pa1 = {"x": x1, "y": y1, "kills": 0, "items": 0, "damage_taken": 0.0, "dead": False, "finished": False}
        pb0 = {**pa0, "x": x0 + shift_x, "y": y0 + shift_y}
        pb1 = {**pa1, "x": x1 + shift_x, "y": y1 + shift_y}
        ra = sa.step(pa0, pa1, fwd)
        rb = sb.step(pb0, pb1, fwd)
        assert ra == pytest.approx(rb, abs=1e-6)


def test_idle_wall_shot_and_weapon_swap_penalties() -> None:
    shaper = RewardShaper({"empty_wall_shot": -0.08, "unnecessary_weapon_swap": -0.06})
    shaper.reset([])
    p0 = {"x": 0.0, "y": 0.0, "kills": 0, "items": 0, "damage_taken": 0.0, "dead": False, "finished": False}
    p1 = {"x": 0.0, "y": 0.0, "kills": 0, "items": 0, "damage_taken": 0.0, "dead": False, "finished": False}
    act_shoot = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    act_wpn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)

    shaper.step(p0, p1, act_shoot, info={"enemy_visible": False, "has_ammo": True})
    assert shaper.last_components.get("empty_wall_shot") == pytest.approx(-0.08, abs=1e-6)

    # Damaging non-lethal shot when enemy IS visible must NOT be penalized
    shaper.step(p0, p1, act_shoot, info={"enemy_visible": True, "has_ammo": True})
    assert shaper.last_components.get("empty_wall_shot") == pytest.approx(0.0, abs=1e-6)

    # Omitted info must NOT trigger empty_wall_shot penalty
    shaper.step(p0, p1, act_shoot, info=None)
    assert shaper.last_components.get("empty_wall_shot") == pytest.approx(0.0, abs=1e-6)

    # Unnecessary weapon swap when holding ammo IS penalized
    shaper.step(p0, p1, act_wpn, info={"enemy_visible": False, "has_ammo": True})
    assert shaper.last_components.get("unnecessary_weapon_swap") == pytest.approx(-0.06, abs=1e-6)

    # Weapon swap when out of ammo is NOT penalized
    shaper.step(p0, p1, act_wpn, info={"enemy_visible": False, "has_ammo": False})
    assert shaper.last_components.get("unnecessary_weapon_swap") == pytest.approx(0.0, abs=1e-6)
