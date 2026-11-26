"""Comprehensive interface, invariance, latency, and integration tests for Milestone 2 (M2)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from doom_env import make_env
from doom_env.sensors import RAY_DOOR, RAY_LOCKED
from fixtures import fixture_wad
from training.curriculum_plr import PLRBuffer
from training.mapgen import ALL_SCENARIOS, MapGen
from training.model import HybridDoomNet, RecurrentState
from training.preprocessing import (
    ModelInputs,
    OdomGridState,
    preprocess,
    stack_model_inputs,
    update_action_state,
)
from training.rewards import RewardShaper, attribute_sector_rewards

ROOT = Path(__file__).resolve().parent.parent


def _synthetic_obs(h: int = 48, w: int = 64, *, weapon: int = 2, ammo: int = 10) -> dict:
    rng = np.random.default_rng(7)
    rays_dist = rng.uniform(0.1, 0.9, size=(64,)).astype(np.float32)
    rays_type = np.ones((64,), dtype=np.int64)
    vitals = np.zeros((17,), dtype=np.float32)
    vitals[0] = 100.0
    vitals[1] = 25.0
    vitals[2] = float(weapon)
    vitals[3] = float(ammo)
    vitals[4] = float(ammo)
    vitals[14] = 8.0  # vel_forward
    return {
        "rgb": rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        "depth": rng.integers(0, 256, size=(h, w), dtype=np.uint8),
        "vitals": vitals,
        "rays": {"distance": rays_dist, "hit_type": rays_type},
        "visible_entities": {
            "features": np.zeros((32, 5), dtype=np.float32),
            "entity_type": np.zeros((32,), dtype=np.int64),
            "mask": np.zeros((32,), dtype=np.int8),
        },
    }


# --------------------------------------------------------------------------- 1. Configs
def test_yaml_configs_schema_and_limits():
    ppo = yaml.safe_load((ROOT / "training/configs/ppo_curriculum.yaml").read_text())
    bc = yaml.safe_load((ROOT / "training/configs/bc.yaml").read_text())
    ev = yaml.safe_load((ROOT / "training/configs/eval.yaml").read_text())

    # 100% self-evolving RL primary loop + value compass
    assert ppo["bc_kl_beta_init"] == 0.0
    assert ppo["bc_enabled"] is False
    assert ppo["compass_coef"] == pytest.approx(0.15)
    assert ppo["aux_coef"] == pytest.approx(0.10)
    assert ppo["num_envs"] == 16
    assert ppo["rollout_steps"] == 128
    assert ppo["bptt_len"] == 32
    assert ppo["tier_weights"] == [1.0, 1.5, 2.5, 3.0]
    assert ppo["cpu_threads"] == 1
    assert ppo["max_vram_mb"] == 2048
    assert ppo["max_ram_mb"] == 8192
    assert ppo["agent_ms_per_step"] == 40.0
    assert ppo["plr"]["seed_min"] >= 100_000

    assert bc["optional"] is True
    assert bc["seq_len"] == 32
    assert bc["pos_weight_use"] == pytest.approx(2.5)
    assert bc["pos_weight_attack"] == pytest.approx(2.5)
    assert bc["model_config"]["gru_dim"] == 256

    assert ev["device"] == "cpu"
    assert ev["cpu_threads"] == 1
    assert ev["max_vram_mb"] == 2048
    assert ev["max_ram_mb"] == 8192
    assert ev["agent_ms_per_step"] == 40.0
    assert ev["latency_target_ms"] == 25.0
    assert ev["heldout_seed_max"] < ppo["plr"]["seed_min"]


# --------------------------------------------------------------------------- 2. Preprocessing
def test_preprocess_shapes_both_resolutions_and_entity_scaling():
    for h, w in ((120, 160), (48, 64)):
        obs = _synthetic_obs(h, w)
        obs["visible_entities"]["features"][0] = [0.25, 0.0, 1.0, 30.0, -15.0]
        obs["visible_entities"]["entity_type"][0] = 2
        obs["visible_entities"]["mask"][0] = 1
        # Garbage in padded slot 1 must be zeroed out
        obs["visible_entities"]["features"][1] = [999.0, 1.0, 1.0, 999.0, 999.0]

        inp, st = preprocess(obs, None)
        assert isinstance(inp, ModelInputs)
        assert isinstance(st, OdomGridState)
        assert inp.rays.shape == (1, 8, 64) and inp.rays.dtype == torch.float32
        assert inp.ent_feats.shape == (1, 32, 5) and inp.ent_feats.dtype == torch.float32
        assert inp.ent_types.shape == (1, 32) and inp.ent_types.dtype == torch.int64
        assert inp.ent_mask.shape == (1, 32) and inp.ent_mask.dtype == torch.bool
        assert inp.vitals.shape == (1, 36) and inp.vitals.dtype == torch.float32
        assert inp.rgb_depth.shape == (1, 4, 48, 64) and inp.rgb_depth.dtype == torch.float32
        assert inp.grid.shape == (1, 6, 32, 32) and inp.grid.dtype == torch.float32
        assert inp.action_mask.shape == (1, 9) and inp.action_mask.dtype == torch.bool

        assert inp.ent_feats[0, 0, 3].item() == pytest.approx(1.0)
        assert inp.ent_feats[0, 0, 4].item() == pytest.approx(-0.5)
        assert torch.all(inp.ent_feats[0, 1] == 0.0)


def test_preprocess_reset_and_action_masks():
    obs_door = _synthetic_obs(48, 64, weapon=2, ammo=0)
    obs_door["rays"]["hit_type"][0] = RAY_DOOR
    obs_door["rays"]["distance"][0] = 0.08

    # Weapon > 1 with 0 ammo masks ATTACK (idx 6)
    inp0, st = preprocess(obs_door, None)
    assert bool(inp0.action_mask[0, 6]) is False
    assert bool(inp0.action_mask[0, 7]) is True

    # Fist/chainsaw (weapon 0 and 1) allow ATTACK even at 0 ammo
    for melee_w in (0, 1):
        obs_melee = _synthetic_obs(48, 64, weapon=melee_w, ammo=0)
        inp_m, _ = preprocess(obs_melee, None)
        assert bool(inp_m.action_mask[0, 6]) is True

    # Trigger USE facing door -> cooldown = 6 steps
    use_act = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int8)
    st = update_action_state(st, use_act, obs_door)
    assert st.use_cooldown == 6

    for _ in range(6):
        inp_cd, st = preprocess(obs_door, st)
        assert bool(inp_cd.action_mask[0, 7]) is False

    # Step 7: USE is allowed again
    inp_after, st = preprocess(obs_door, st)
    assert bool(inp_after.action_mask[0, 7]) is True

    # Clean reset when state=None
    inp_a, _ = preprocess(obs_door, None)
    inp_b, _ = preprocess(obs_door, None)
    torch.testing.assert_close(inp_a.grid, inp_b.grid)
    torch.testing.assert_close(inp_a.vitals, inp_b.vitals)


def test_preprocess_translation_invariance_in_doom_env():
    grid = "#########\n#P.....z#\n#..D....#\n#########"
    shifted = "\n".join(["#" * 13] * 3 + ["####" + row for row in grid.splitlines()])
    env_a = make_env(fixture_wad(grid), render_buffers=False)
    env_b = make_env(fixture_wad(shifted), render_buffers=False)
    try:
        oa, _ = env_a.reset(seed=3)
        ob, _ = env_b.reset(seed=3)
        sa = sb = None
        for step in range(8):
            ia, sa = preprocess(oa, sa)
            ib, sb = preprocess(ob, sb)
            torch.testing.assert_close(ia.rays, ib.rays, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(ia.vitals, ib.vitals, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(ia.grid, ib.grid, atol=1e-4, rtol=1e-4)
            act = np.array([1 if step % 2 == 0 else 0, 0, 0, 0, 1 if step % 2 else 0, 0, 0, 0, 0], np.int8)
            sa = update_action_state(sa, act, oa)
            sb = update_action_state(sb, act, ob)
            oa, *_ = env_a.step(act)
            ob, *_ = env_b.step(act)
    finally:
        env_a.close()
        env_b.close()


# --------------------------------------------------------------------------- 3. HybridDoomNet
def test_model_forward_zero_entities_nan_safe_and_aux_heads():
    net = HybridDoomNet()
    net.train()
    bsz = 3
    obs = _synthetic_obs(48, 64, weapon=2, ammo=0)
    inp_single, _ = preprocess(obs, None)
    batch = stack_model_inputs([inp_single] * bsz, dim=0)
    # Explicitly verify all-zero entity mask produces zero NaNs forward AND backward
    batch.ent_mask.zero_()
    h0 = net.init_state(bsz)

    logits, value, aux, next_h = net(batch, h0, return_aux=True)
    assert logits.shape == (bsz, 9)
    assert value.shape == (bsz,)
    assert isinstance(aux, dict)
    assert aux["next_rays"].shape == (bsz, 64)
    assert aux["value_compass"].shape == (bsz, 8)
    assert aux["obj_slots"].shape == (bsz, 4, 8)
    assert aux["door_prox"].shape == (bsz, 3)
    assert aux["key_prox"].shape == (bsz, 3)
    assert aux["exit_prox"].shape == (bsz, 3)
    assert next_h.gru_h.shape == (1, bsz, 256)
    assert next_h.obj_slots.shape == (bsz, 4, 8)

    # Masked ATTACK (idx 6) must be -1e8
    assert torch.all(logits[:, 6] <= -1e7)

    loss = (
        logits.sigmoid().sum()
        + value.pow(2).sum()
        + aux["value_compass"].pow(2).sum()
        + aux["next_rays"].sum()
        + aux["door_prox"].pow(2).sum()
        + aux["key_prox"].pow(2).sum()
        + aux["exit_prox"].pow(2).sum()
    )
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"

    logits_noaux, val_noaux, aux_none, _ = net(batch, h0, return_aux=False)
    assert aux_none is None
    assert logits_noaux.shape == (bsz, 9) and val_noaux.shape == (bsz,)


def test_model_bptt_sequence_done_masking():
    net = HybridDoomNet()
    net.eval()
    bsz, seq_len = 2, 6
    inp, _ = preprocess(_synthetic_obs(), None)
    single = stack_model_inputs([inp] * bsz, dim=0)
    seq_dict = {k: v.unsqueeze(1).expand(bsz, seq_len, *v.shape[1:]).clone() for k, v in single.as_dict().items()}
    done_mask = torch.zeros((bsz, seq_len), dtype=torch.float32)
    done_mask[0, 2] = 1.0  # env 0 resets after step t=2 -> t=3 starts from zero hidden state

    h0 = RecurrentState(
        gru_h=torch.randn(1, bsz, 256),
        obj_slots=torch.randn(bsz, 4, 8),
    )
    logits_seq, val_seq, aux_seq, _ = net.forward_sequence(seq_dict, h0, done_mask=done_mask, return_aux=True)
    assert logits_seq.shape == (bsz, seq_len, 9)
    assert val_seq.shape == (bsz, seq_len)
    assert aux_seq is not None and aux_seq["value_compass"].shape == (bsz, seq_len, 8)

    # Compare step t=3 of env 0 against a fresh zero-state single-step forward
    fresh_h = net.init_state(1)
    step3_in = {k: v[0:1, 3] for k, v in seq_dict.items()}
    l_fresh, v_fresh, _, _ = net(step3_in, fresh_h, return_aux=False)
    torch.testing.assert_close(logits_seq[0:1, 3], l_fresh, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(val_seq[0:1, 3], v_fresh, atol=1e-5, rtol=1e-5)


def test_single_thread_cpu_inference_latency():
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        net = HybridDoomNet()
        net.eval()
        obs = _synthetic_obs(120, 160)
        state = None
        h = net.init_state(1)
        # Warmup
        for _ in range(10):
            inp, state = preprocess(obs, state)
            with torch.inference_mode():
                _, _, _, h = net(inp, h, return_aux=False)

        t0 = time.perf_counter()
        n_steps = 200
        for _ in range(n_steps):
            inp, state = preprocess(obs, state)
            with torch.inference_mode():
                _, _, _, h = net(inp, h, return_aux=False)
        mean_ms = ((time.perf_counter() - t0) * 1000.0) / n_steps
        assert mean_ms < 40.0, f"Exceeded 40ms step budget: {mean_ms:.2f} ms"
        assert mean_ms < 25.0, f"Exceeded 25ms internal target: {mean_ms:.2f} ms"
    finally:
        torch.set_num_threads(prev_threads)


# --------------------------------------------------------------------------- 4. Rewards
def test_reward_shaper_deltas_and_translation_invariance():
    # Two disjoint rectangular sectors (0: [0,200]x[0,200], 1: [200,400]x[0,200])
    polys = [
        (0, np.array([[0, 0, 0, 200], [0, 200, 200, 200], [200, 200, 200, 0], [200, 0, 0, 0]], np.float64)),
        (1, np.array([[200, 0, 200, 200], [200, 200, 400, 200], [400, 200, 400, 0], [400, 0, 200, 0]], np.float64)),
    ]
    shift_x, shift_y = 1337.0, -900.0
    polys_shifted = [
        (sid, segs + np.array([shift_x, shift_y, shift_x, shift_y], np.float64))
        for sid, segs in polys
    ]

    sh_a = RewardShaper()
    sh_b = RewardShaper()
    sh_a.reset(polys)
    sh_b.reset(polys_shifted)

    priv_prev = {
        "tic": 4, "episode_ticks": 2100, "x": 100.0, "y": 100.0, "angle": 0.0,
        "kills": 0, "items": 0, "damage_taken": 0.0, "keys": np.zeros(6, np.float32),
        "dead": False, "exit_reached": False,
        "sector_floor": np.array([0.0, 0.0, 0.0]),
        "sector_ceiling": np.array([128.0, 128.0, 0.0]),
        "door_lock": np.array([-1, -1, 129]),
    }
    priv_now = {
        "tic": 8, "episode_ticks": 2100, "x": 250.0, "y": 100.0, "angle": 0.0,
        "kills": 1, "items": 1, "damage_taken": 10.0, "keys": np.array([0, 0, 1, 0, 0, 0], np.float32),
        "dead": False, "exit_reached": True,
        "sector_floor": np.array([0.0, 0.0, 0.0]),
        "sector_ceiling": np.array([128.0, 128.0, 32.0]),
        "door_lock": np.array([-1, -1, 129]),
    }
    priv_prev_s = {**priv_prev, "x": priv_prev["x"] + shift_x, "y": priv_prev["y"] + shift_y}
    priv_now_s = {**priv_now, "x": priv_now["x"] + shift_x, "y": priv_now["y"] + shift_y}

    act = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    r_a = sh_a.step(priv_prev, priv_now, act, {"exit_reached": True})
    r_b = sh_b.step(priv_prev_s, priv_now_s, act, {"exit_reached": True})

    assert r_a == pytest.approx(r_b, abs=1e-7)
    assert r_a > 20.0  # exit + sector + key + locked_door + kill + item - damage - step
    assert sh_a.last_components["locked_door"] == pytest.approx(5.0)
    assert sh_a.last_components["key"] == pytest.approx(4.0)

    # Ego-directional sector attribution across headings
    c0 = attribute_sector_rewards({"x": 0.0, "y": 0.0, "angle": 0.0}, {"x": 10.0, "y": 0.0}, 1.0)
    assert int(np.argmax(c0)) == 0  # straight ahead
    c2 = attribute_sector_rewards({"x": 0.0, "y": 0.0, "angle": 0.0}, {"x": 0.0, "y": 10.0}, 1.0)
    assert int(np.argmax(c2)) == 2  # +90 deg left
    c6 = attribute_sector_rewards({"x": 0.0, "y": 0.0, "angle": 90.0}, {"x": 10.0, "y": 0.0}, 1.0)
    assert int(np.argmax(c6)) == 6  # facing North (+90), moving East (+x) is right (-90 deg -> bin 6)


# --------------------------------------------------------------------------- 5. MapGen & doom_env
def test_mapgen_all_tiers_and_extreme_scenarios_load_in_doom_env(tmp_path):
    origins = set()
    for tier in (1, 2, 3, 4):
        for idx, scenario in enumerate(ALL_SCENARIOS):
            seed = 100_000 + tier * 100 + idx
            wad_path, map_name = MapGen.sample(
                tier=tier,
                knobs={"scenario": scenario, "out_dir": str(tmp_path)},
                seed=seed,
                scaffolding_alpha=0.0,
            )
            assert wad_path.is_file()
            assert map_name == "MAP01"

            env = make_env(wad_path, map_name, episode_ticks=200, render_buffers=False)
            try:
                obs, info = env.reset(seed=seed)
                assert env.observation_space.contains(obs)
                assert env._geometry.source == "wad"
                assert len(env.privileged_state()["sector_floor"]) < 2048
                origins.add((round(env.privileged_state()["x"], -1), round(env.privileged_state()["y"], -1)))
                obs, _, term, trunc, info = env.step(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8))
                assert env.observation_space.contains(obs)
            finally:
                env.close()
    assert len(origins) > 5, "Domain randomization must vary global map origins"


def test_mapgen_doors_locked_doors_and_exit_detection(tmp_path):
    # Multi-room doors -> must expose RAY_DOOR (2)
    wad_door, m_door = MapGen.sample(
        tier=2,
        knobs={"scenario": "multi_room_doors", "num_rooms": 4, "out_dir": str(tmp_path)},
        seed=123456,
        scaffolding_alpha=0.0,
    )
    env_d = make_env(wad_door, m_door, render_buffers=False)
    try:
        env_d.reset()
        st = env_d.privileged_state()
        assert np.any(st["door_lock"] == 0), "Tier 2 multi_room_doors must include normal doors"
    finally:
        env_d.close()

    # Keycard progression -> must expose locked door (door_lock > 0) and key entity
    wad_lock, m_lock = MapGen.sample(
        tier=3,
        knobs={"scenario": "keycard_progression", "num_rooms": 5, "out_dir": str(tmp_path)},
        seed=234567,
        scaffolding_alpha=0.0,
    )
    env_l = make_env(wad_lock, m_lock, render_buffers=False)
    try:
        env_l.reset()
        st = env_l.privileged_state()
        assert np.any(st["door_lock"] > 0), "Tier 3 keycard_progression must include locked doors"
        # Verify at least one door sector in mapinfo has lock > 0
        assert any(lk > 0 for lk in env_l.mapinfo.door_sectors.values())
    finally:
        env_l.close()

    # Verify a generated 2-room open_arena reaches exit when navigating across it
    wad_open, m_open = MapGen.sample(
        tier=1,
        knobs={"scenario": "open_arena", "num_rooms": 1, "enemies_per_room": 0, "origin_x": 512, "origin_y": -512, "rot_k": 0, "cell_size": 192, "out_dir": str(tmp_path)},
        seed=345678,
        scaffolding_alpha=0.0,
    )
    env_o = make_env(wad_open, m_open, episode_ticks=400, render_buffers=False)
    try:
        obs, _ = env_o.reset()
        # Verify ray types work cleanly and agent can move without geometry snagging
        reached = False
        for step in range(80):
            act = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
            obs, _, term, trunc, info = env_o.step(act)
            if info["exit_reached"]:
                reached = True
                break
            if term or trunc:
                break
        # Walk straight forward (rot_k=0 places exit alcove on +x wall directly ahead)
        assert reached is True, "Player walking straight across 1-room MapGen arena must reach exit alcove"
    finally:
        env_o.close()


def test_mapgen_rays_detect_door_and_locked_door_types(tmp_path):
    # Directly verify a small generated map with door_lock == 0 and door_lock > 0 emits RAY_DOOR and RAY_LOCKED
    # by testing both MapGen's compiled sectors and doom_env's LevelGeometry raycaster
    wad_path, map_name = MapGen.sample(
        tier=3,
        knobs={"scenario": "keycard_progression", "num_rooms": 4, "enemies_per_room": 0, "cell_size": 192, "out_dir": str(tmp_path)},
        seed=456789,
        scaffolding_alpha=0.0,
    )
    env = make_env(wad_path, map_name, render_buffers=False)
    try:
        env.reset()
        geo = env._geometry
        assert np.any(geo.hit_type == RAY_LOCKED), "Generated Tier 3 map must have RAY_LOCKED linedefs"
    finally:
        env.close()

    wad_d, map_d = MapGen.sample(
        tier=2,
        knobs={"scenario": "multi_room_doors", "num_rooms": 4, "enemies_per_room": 0, "cell_size": 192, "out_dir": str(tmp_path)},
        seed=567890,
        scaffolding_alpha=0.0,
    )
    env2 = make_env(wad_d, map_d, render_buffers=False)
    try:
        env2.reset()
        assert np.any(env2._geometry.hit_type == RAY_DOOR), "Generated Tier 2 map must have RAY_DOOR linedefs"
    finally:
        env2.close()


# --------------------------------------------------------------------------- 6. PLRBuffer
def test_plr_buffer_seeds_priority_and_knobs():
    plr = PLRBuffer(capacity=8, seed=11)
    assert plr.scaffolding_alpha(0) == pytest.approx(1.0)
    assert plr.scaffolding_alpha(6_000_000) == pytest.approx(0.5)
    assert plr.scaffolding_alpha(12_000_000) == pytest.approx(0.0)
    assert plr.scaffolding_alpha(20_000_000) == pytest.approx(0.0)

    seen_seeds = set()
    for step in range(20):
        lvl = plr.sample_level(global_step=step * 100_000, update_idx=step)
        assert 100_000 <= lvl["seed"] <= 999_999
        seen_seeds.add(lvl["seed"])
        plr.update_score(lvl["level_id"], gae_score=0.2 + 0.1 * (step % 5), update_idx=step)

    assert len(plr) <= 8
    assert all(100_000 <= s <= 999_999 for s in seen_seeds)

    updated = plr.update_knobs({"enemies_per_room_delta": 10.0, "ammo_mult_delta": -5.0, "tier_weights": [0.5, 1.0, 2.0, 4.0]})
    assert updated["global_knobs"]["enemies_per_room_mult"] <= 2.0
    assert updated["global_knobs"]["ammo_mult"] >= 0.3
    assert updated["tier_weights"] == [0.5, 1.0, 2.0, 4.0]


# --------------------------------------------------------------------------- 7. Full M2 End-to-End Loop
def test_m2_end_to_end_pipeline_zero_nans(tmp_path):
    plr = PLRBuffer(capacity=16, seed=77)
    lvl = plr.sample_level(global_step=0, update_idx=0)
    knobs = {**lvl["knobs"], "out_dir": str(tmp_path)}
    wad_path, map_name = MapGen.sample(
        tier=lvl["tier"],
        knobs=knobs,
        seed=lvl["seed"],
        scaffolding_alpha=lvl["scaffolding_alpha"],
    )

    env = make_env(wad_path, map_name, episode_ticks=200, render_buffers=True)
    net = HybridDoomNet()
    net.eval()
    shaper = RewardShaper()
    try:
        obs, _ = env.reset(seed=lvl["seed"])
        shaper.reset(env.sector_polygons())
        priv_prev = env.privileged_state()
        odom_state = None
        h = net.init_state(1)

        total_shaped = 0.0
        for _ in range(10):
            inp, odom_state = preprocess(obs, odom_state)
            with torch.inference_mode():
                logits, value, aux, h = net(inp, h, return_aux=True)
            assert torch.isfinite(logits).all()
            assert torch.isfinite(value).all()
            assert torch.isfinite(aux["value_compass"]).all()
            assert torch.isfinite(aux["next_rays"]).all()

            probs = torch.sigmoid(logits[0]).numpy()
            action = (probs > 0.5).astype(np.int8)
            odom_state = update_action_state(odom_state, action, obs)
            obs, _, term, trunc, info = env.step(action)
            priv_now = env.privileged_state()
            shaped = shaper.step(priv_prev, priv_now, action, info)
            compass_tgt = attribute_sector_rewards(priv_prev, priv_now, shaper.last_components)
            assert np.isfinite(shaped)
            assert compass_tgt.shape == (8,) and np.isfinite(compass_tgt).all()
            total_shaped += shaped
            priv_prev = priv_now
            if term or trunc:
                break

        plr.update_score(lvl["level_id"], abs(total_shaped), update_idx=1)
    finally:
        env.close()
