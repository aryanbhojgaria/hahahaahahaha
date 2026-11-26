"""Custom Self-Evolving Recurrent PPO with GAE, Truncated BPTT, RND, Value Compass & Optional BC-KL.

Written from scratch in PyTorch per `planning/Design.md` and `planning/Schema.md`:
- Strictly 100% self-evolving RL from scratch (zero human demos required).
- Truncated BPTT (`bptt_len`) with `(1 - done[t-1])` hidden-state + objective-slot masking.
- Observation-masked 9-button Bernoulli policy & entropy calculation.
- Auxiliary supervision for `next_rays` and 8-sector `value_compass` ("plus-point" finder).
- Optional decaying BC-KL anchoring when `bc_anchor_checkpoint` & `bc_kl_beta_init > 0` are set.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from doom_env import CONTRACT_VERSION
from training.curriculum_plr import PLRBuffer
from training.envs import VecDoomEnv
from training.mapgen.tiers import MapGen
from training.model import HybridDoomNet, RecurrentState
from training.preprocessing import ModelInputs, OdomGridState, preprocess, update_action_state
from training.rewards import attribute_sector_rewards
from training.rnd import RNDModule, compute_grid_novelty


def _current_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation over `(T, B)` tensors."""
    t_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    gae = torch.zeros_like(next_value, dtype=torch.float32)

    for t in reversed(range(t_steps)):
        next_val = next_value if t == t_steps - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t].float()
        delta = rewards[t].float() + float(gamma) * next_val.float() * nonterminal - values[t].float()
        gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
        advantages[t] = gae

    returns = advantages + values.float()
    return advantages, returns


def masked_bernoulli_log_prob_entropy(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked MultiBinary(9) Bernoulli log-probability and entropy.

    Masked button indices (`mask == False`) contribute strictly `0.0` to log_prob, entropy, and grad.
    """
    mask_b = action_mask.bool()
    mask_f = mask_b.to(dtype=torch.float32)
    safe_logits = torch.where(mask_b, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    safe_targets = (actions.float() * mask_f).clamp(0.0, 1.0)

    bce = F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none")
    log_prob = -(bce * mask_f).sum(dim=-1)

    probs = torch.sigmoid(safe_logits).clamp(1e-6, 1.0 - 1e-6)
    per_btn_ent = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    entropy = (per_btn_ent * mask_f).sum(dim=-1)
    return log_prob, entropy


masked_bernoulli = masked_bernoulli_log_prob_entropy


def _stack_list_to_bt(items: Sequence[ModelInputs], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert `T`-length list of `(B, ...)` ModelInputs into `(B, T, ...)` dict on `device`."""
    keys = ("rays", "ent_feats", "ent_types", "ent_mask", "vitals", "rgb_depth", "grid", "action_mask")
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        # Each item tensor has shape (B, ...); stack on dim=1 -> (B, T, ...)
        out[k] = torch.stack([getattr(it, k).to(device) for it in items], dim=1)
    if items[0].d_theta is not None:
        out["d_theta"] = torch.stack([it.d_theta.to(device) for it in items], dim=1)
    return out


def bptt_unroll(
    model: HybridDoomNet,
    inputs_seq: Sequence[ModelInputs] | dict[str, torch.Tensor],
    dones: torch.Tensor,
    init_h: Optional[RecurrentState] = None,
    return_aux: bool = False,
    rgb_dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]], RecurrentState]:
    """Unroll `model.forward_sequence` over a `T`-step chunk with `(1 - done[t-1])` reset masking.

    When `inputs_seq` is a `Sequence[ModelInputs]` of length `T`, `dones` is `(T, B)` and outputs
    are returned time-major `(T, B, ...)` so `logits[t]` aligns directly with `inputs_seq[t]`.
    """
    device = next(model.parameters()).device
    time_major = isinstance(inputs_seq, (list, tuple))
    if time_major:
        seq_len = len(inputs_seq)
        bsz = inputs_seq[0].rays.shape[0]
        seq_dict = _stack_list_to_bt(inputs_seq, device)
        dones_bt = dones.t() if (dones.ndim == 2 and dones.shape == (seq_len, bsz)) else dones
    else:
        seq_dict = {k: v.to(device) for k, v in inputs_seq.items()}
        bsz, seq_len = seq_dict["rays"].shape[:2]
        dones_bt = dones

    logits_bt, values_bt, aux_bt, final_h = model.forward_sequence(
        seq_dict,
        h=init_h.to(device) if init_h is not None else None,
        done_mask=dones_bt.to(device),
        return_aux=return_aux,
        rgb_dropout_p=rgb_dropout_p,
    )

    if time_major:
        logits = logits_bt.transpose(0, 1).contiguous()
        values = values_bt.transpose(0, 1).contiguous()
        aux_out = (
            {k: v.transpose(0, 1).contiguous() for k, v in aux_bt.items()}
            if aux_bt is not None
            else None
        )
        return logits, values, aux_out, final_h

    return logits_bt, values_bt, aux_bt, final_h


unroll_bptt_chunk = bptt_unroll


def save_checkpoint(
    path: str | Path,
    *,
    model: HybridDoomNet,
    model_config: Optional[dict[str, Any]] = None,
    global_step: int = 0,
    update_idx: int = 0,
    scorer_summary: Optional[dict[str, Any]] = None,
    obs_norm_stats: Optional[dict[str, Any]] = None,
) -> Path:
    """Save a Schema.md §4 compliant `.pt` checkpoint."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_version": str(CONTRACT_VERSION),
        "git_sha": _current_git_sha(),
        "global_step": int(global_step),
        "update_idx": int(update_idx),
        "model_config": dict(model_config or model.model_config),
        "model_state_dict": model.state_dict(),
        "obs_norm": obs_norm_stats or {},
        "scorer_summary": dict(
            scorer_summary or {"total_score": 0.0, "mean_act_ms": 0.0, "p99_act_ms": 0.0}
        ),
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a Schema.md §4 `.pt` checkpoint, raising FileNotFoundError if missing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    payload = torch.load(p, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Invalid checkpoint payload in {p}")
    return payload


def _rnd_feature_vector(inp: ModelInputs) -> torch.Tensor:
    """Concatenate rays (B, 8*64) + vitals (B, 36) -> (B, 548) for RND curiosity."""
    return torch.cat([inp.rays.flatten(1), inp.vitals], dim=-1)


def _bernoulli_kl(logits_p: torch.Tensor, logits_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(Bernoulli(p) || Bernoulli(q)) masked by `mask`, summed over action dim."""
    mask_f = mask.bool().to(dtype=torch.float32)
    p = torch.sigmoid(logits_p.float()).clamp(1e-6, 1.0 - 1e-6)
    q = torch.sigmoid(logits_q.float()).clamp(1e-6, 1.0 - 1e-6)
    kl = p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (torch.log(1.0 - p) - torch.log(1.0 - q))
    return (kl * mask_f).sum(dim=-1)


def _resolve_device(dev_spec: str) -> torch.device:
    if dev_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_spec)


def _load_cfg_dict(cfg_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg_or_path, dict):
        return dict(cfg_or_path)
    p = Path(cfg_or_path)
    if not p.is_file():
        raise FileNotFoundError(f"PPO config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML dict in {p}")
    return data


def train(cfg_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Run 100% self-evolving Recurrent PPO (+ RND + Value Compass + PLR) inside ViZDoom."""
    cfg = _load_cfg_dict(cfg_or_path)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    device = _resolve_device(str(cfg.get("device", "auto")))
    num_envs = int(cfg.get("num_envs", 16))
    rollout_steps = int(cfg.get("rollout_steps", 128))
    bptt_len = int(cfg.get("bptt_len", 32))
    if rollout_steps % bptt_len != 0:
        raise ValueError(f"rollout_steps ({rollout_steps}) must be divisible by bptt_len ({bptt_len})")

    ppo_epochs = int(cfg.get("ppo_epochs", 4))
    minibatch_chunks = int(cfg.get("minibatch_chunks", 16))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_coef = float(cfg.get("clip_coef", 0.2))
    lr = float(cfg.get("lr", 2.5e-4))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    vf_coef = float(cfg.get("vf_coef", 0.5))
    ent_coef = float(cfg.get("ent_coef", 0.01))
    rnd_coef = float(cfg.get("rnd_coef", 0.05))
    grid_novelty_coef = float(cfg.get("grid_novelty_coef", 0.03))
    aux_coef = float(cfg.get("aux_coef", 0.10))
    compass_coef = float(cfg.get("compass_coef", 0.15))
    bc_kl_beta_init = float(cfg.get("bc_kl_beta_init", 0.0))
    bc_kl_decay_steps = max(1, int(cfg.get("bc_kl_decay_steps", 4_000_000)))
    rgb_dropout_p = float(cfg.get("rgb_dropout_p", 0.25))
    total_updates = int(cfg.get("total_updates", 10))
    save_interval = max(1, int(cfg.get("save_interval", max(1, total_updates))))
    episode_ticks = int(cfg.get("episode_ticks", 2100))

    output_dir = Path(cfg.get("output_dir", "runs/ppo_default"))
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = dict(
        cfg.get(
            "model_config",
            {
                "d_model": 256,
                "gru_dim": 256,
                "num_slots": 4,
                "slot_dim": 8,
                "use_rgb": True,
                "use_grid": True,
            },
        )
    )
    model = HybridDoomNet(model_cfg).to(device)
    global_step = 0
    start_update = 0

    resume_path = cfg.get("resume_checkpoint")
    if resume_path:
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        global_step = int(ckpt.get("global_step", 0))
        start_update = int(ckpt.get("update_idx", 0))

    bc_anchor: Optional[HybridDoomNet] = None
    bc_anchor_path = cfg.get("bc_anchor_checkpoint")
    if bc_anchor_path and bc_kl_beta_init > 0.0:
        bc_ckpt = load_checkpoint(bc_anchor_path, map_location=device)
        bc_anchor = HybridDoomNet(bc_ckpt.get("model_config", model_cfg)).to(device)
        bc_anchor.load_state_dict(bc_ckpt["model_state_dict"])
        bc_anchor.eval()
        for p in bc_anchor.parameters():
            p.requires_grad = False

    rnd = RNDModule(feature_dim=8 * 64 + 36).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(rnd.predictor.parameters()),
        lr=lr,
        eps=1e-5,
    )

    plr_cfg = dict(cfg.get("plr", {}))
    plr = PLRBuffer(
        capacity=int(plr_cfg.get("buffer_size", 256)),
        tier_weights=cfg.get("tier_weights", [1.0, 1.5, 2.5, 3.0]),
        seed_min=int(plr_cfg.get("seed_min", 100_000)),
        seed_max=int(plr_cfg.get("seed_max", 999_999)),
        replay_prob=float(plr_cfg.get("replay_prob", 0.5)),
        staleness_coef=float(plr_cfg.get("staleness_coef", 0.1)),
        temperature=float(plr_cfg.get("temperature", 0.1)),
        scaffolding_end_step=int(cfg.get("scaffolding_end_step", 12_000_000)),
        seed=seed,
    )

    fixed_wad = cfg.get("wad_path")
    wads_dir = output_dir / "wads"
    wads_dir.mkdir(parents=True, exist_ok=True)
    if fixed_wad:
        init_wads = [str(fixed_wad)] * num_envs
        env_levels = [None] * num_envs
    else:
        init_wads = []
        env_levels = []
        for _ in range(num_envs):
            lvl = plr.sample_level(global_step=global_step, update_idx=start_update)
            knobs = {**lvl["knobs"], "out_dir": str(wads_dir)}
            w_path, _ = MapGen.sample(
                tier=lvl["tier"],
                knobs=knobs,
                seed=lvl["seed"],
                scaffolding_alpha=lvl["scaffolding_alpha"],
            )
            init_wads.append(str(w_path))
            env_levels.append(lvl)

    venv = VecDoomEnv(
        num_envs=num_envs,
        wads=init_wads,
        episode_ticks=episode_ticks,
        render_buffers_ratio=float(cfg.get("render_buffers_ratio", 0.25)),
        worker_max_episodes=int(cfg.get("worker_max_episodes", 100)),
        worker_timeout_s=float(cfg.get("worker_timeout_s", 15.0)),
        use_reward_shaper=True,
        seed=seed,
    )

    wandb_run = None
    if bool(cfg.get("wandb_enabled", False)):
        try:
            import wandb

            wandb_run = wandb.init(project=str(cfg.get("wandb_project", "devforge-doom")), config=cfg)
        except Exception:
            wandb_run = None

    last_loss = 0.0
    last_mean_return = 0.0
    last_ckpt_path: Optional[Path] = None

    try:
        obs_list, info_list = venv.reset()
        odom_states: list[Optional[OdomGridState]] = [None] * num_envs
        prev_privs = [inf.get("privileged_state") for inf in info_list]
        rec_state = model.init_state(num_envs, device=device)

        num_chunks_per_env = rollout_steps // bptt_len

        for upd in range(start_update + 1, start_update + total_updates + 1):
            model.eval()

            buf_inputs: list[ModelInputs] = []
            buf_actions = torch.zeros((rollout_steps, num_envs, 9), dtype=torch.float32)
            buf_log_probs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
            buf_values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
            buf_rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
            buf_dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
            buf_next_rays = torch.zeros((rollout_steps, num_envs, 64), dtype=torch.float32)
            buf_compass_tgt = torch.zeros((rollout_steps, num_envs, 8), dtype=torch.float32)

            chunk_init_gru: list[torch.Tensor] = []
            chunk_init_slots: list[torch.Tensor] = []

            ep_returns = np.zeros(num_envs, dtype=np.float32)
            completed_returns: list[float] = []

            for step in range(rollout_steps):
                if step % bptt_len == 0:
                    chunk_init_gru.append(rec_state.gru_h.detach().cpu())
                    chunk_init_slots.append(rec_state.obj_slots.detach().cpu())

                preps: list[ModelInputs] = []
                new_odoms: list[OdomGridState] = []
                for i in range(num_envs):
                    inp_i, st_i = preprocess(
                        obs_list[i],
                        odom_states[i],
                        device="cpu",
                        use_grid=bool(model_cfg.get("use_grid", True)),
                    )
                    preps.append(inp_i)
                    new_odoms.append(st_i)

                batch_inp = ModelInputs(
                    rays=torch.cat([p.rays for p in preps], dim=0),
                    ent_feats=torch.cat([p.ent_feats for p in preps], dim=0),
                    ent_types=torch.cat([p.ent_types for p in preps], dim=0),
                    ent_mask=torch.cat([p.ent_mask for p in preps], dim=0),
                    vitals=torch.cat([p.vitals for p in preps], dim=0),
                    rgb_depth=torch.cat([p.rgb_depth for p in preps], dim=0),
                    grid=torch.cat([p.grid for p in preps], dim=0),
                    action_mask=torch.cat([p.action_mask for p in preps], dim=0),
                    d_theta=torch.cat([p.d_theta for p in preps if p.d_theta is not None], dim=0),
                )
                buf_inputs.append(batch_inp)

                with torch.no_grad():
                    batch_dev = batch_inp.to(device)
                    logits, value, _, next_rec = model(batch_dev, rec_state, return_aux=False)
                    probs = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
                    probs = probs * batch_dev.action_mask.float()
                    actions_t = torch.bernoulli(probs)
                    lp, _ = masked_bernoulli_log_prob_entropy(logits, batch_dev.action_mask, actions_t)

                    rnd_feats = _rnd_feature_vector(batch_dev)
                    intr_rnd, _ = rnd.compute_intrinsic_reward(rnd_feats, update_stats=True)
                    intr_grid = compute_grid_novelty(batch_dev.grid)

                actions_np = actions_t.cpu().numpy().astype(np.int8)
                for i in range(num_envs):
                    odom_states[i] = update_action_state(new_odoms[i], actions_np[i], obs_list[i])

                next_obs, extr_rew, term, trunc, next_info = venv.step(actions_np)
                done_np = np.logical_or(term, trunc)

                for i in range(num_envs):
                    buf_next_rays[step, i] = torch.as_tensor(next_obs[i]["rays"]["distance"], dtype=torch.float32)
                    p_prev = prev_privs[i]
                    p_now = next_info[i].get("terminal_privileged_state") or next_info[i].get("privileged_state")
                    if isinstance(p_prev, dict) and isinstance(p_now, dict):
                        comps = next_info[i].get("reward_components", extr_rew[i])
                        buf_compass_tgt[step, i] = torch.from_numpy(
                            attribute_sector_rewards(p_prev, p_now, comps)
                        )
                    ep_returns[i] += float(extr_rew[i])
                    if done_np[i]:
                        completed_returns.append(float(ep_returns[i]))
                        ep_returns[i] = 0.0
                        odom_states[i] = None
                        if not fixed_wad:
                            old_lvl = env_levels[i]
                            if old_lvl is not None:
                                plr.update_score(old_lvl["level_id"], float(abs(extr_rew[i])) + 0.5, upd)
                            new_lvl = plr.sample_level(global_step=global_step, update_idx=upd)
                            knobs = {**new_lvl["knobs"], "out_dir": str(wads_dir)}
                            w_path, _ = MapGen.sample(
                                tier=new_lvl["tier"],
                                knobs=knobs,
                                seed=new_lvl["seed"],
                                scaffolding_alpha=new_lvl["scaffolding_alpha"],
                            )
                            next_obs[i], next_info[i] = venv.reset_with_wad(i, w_path, "MAP01", seed=new_lvl["seed"])
                            env_levels[i] = new_lvl

                    prev_privs[i] = next_info[i].get("privileged_state")

                total_rew = (
                    torch.as_tensor(extr_rew, dtype=torch.float32)
                    + rnd_coef * intr_rnd.cpu()
                    + grid_novelty_coef * intr_grid.cpu()
                )
                done_t = torch.as_tensor(done_np, dtype=torch.float32)

                buf_actions[step] = torch.from_numpy(actions_np.astype(np.float32))
                buf_log_probs[step] = lp.cpu()
                buf_values[step] = value.cpu()
                buf_rewards[step] = total_rew
                buf_dones[step] = done_t

                rec_state = next_rec.detach().mask_done(done_t.to(device))
                obs_list = next_obs
                global_step += num_envs

            # Bootstrap final value
            with torch.no_grad():
                last_preps = [
                    preprocess(
                        obs_list[i],
                        odom_states[i].copy() if odom_states[i] is not None else None,
                        device="cpu",
                        use_grid=bool(model_cfg.get("use_grid", True)),
                    )[0]
                    for i in range(num_envs)
                ]
                last_batch = ModelInputs(
                    rays=torch.cat([p.rays for p in last_preps], dim=0),
                    ent_feats=torch.cat([p.ent_feats for p in last_preps], dim=0),
                    ent_types=torch.cat([p.ent_types for p in last_preps], dim=0),
                    ent_mask=torch.cat([p.ent_mask for p in last_preps], dim=0),
                    vitals=torch.cat([p.vitals for p in last_preps], dim=0),
                    rgb_depth=torch.cat([p.rgb_depth for p in last_preps], dim=0),
                    grid=torch.cat([p.grid for p in last_preps], dim=0),
                    action_mask=torch.cat([p.action_mask for p in last_preps], dim=0),
                    d_theta=torch.cat([p.d_theta for p in last_preps if p.d_theta is not None], dim=0),
                ).to(device)
                _, next_val, _, _ = model(last_batch, rec_state, return_aux=False)

            advantages, returns = compute_gae(
                buf_rewards, buf_values, buf_dones, next_val.cpu(), gamma=gamma, gae_lambda=gae_lambda
            )
            adv_norm = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            # Build (total_chunks, bptt_len, ...) dataset
            total_chunks = num_chunks_per_env * num_envs
            chunk_inputs: list[list[ModelInputs]] = []
            chunk_dones_list: list[torch.Tensor] = []
            chunk_actions_list: list[torch.Tensor] = []
            chunk_old_lp_list: list[torch.Tensor] = []
            chunk_adv_list: list[torch.Tensor] = []
            chunk_ret_list: list[torch.Tensor] = []
            chunk_next_rays_list: list[torch.Tensor] = []
            chunk_compass_list: list[torch.Tensor] = []
            chunk_gru_list: list[torch.Tensor] = []
            chunk_slots_list: list[torch.Tensor] = []

            for c in range(num_chunks_per_env):
                t0 = c * bptt_len
                t1 = t0 + bptt_len
                for env_i in range(num_envs):
                    seq_i = [
                        ModelInputs(
                            rays=buf_inputs[t].rays[env_i : env_i + 1],
                            ent_feats=buf_inputs[t].ent_feats[env_i : env_i + 1],
                            ent_types=buf_inputs[t].ent_types[env_i : env_i + 1],
                            ent_mask=buf_inputs[t].ent_mask[env_i : env_i + 1],
                            vitals=buf_inputs[t].vitals[env_i : env_i + 1],
                            rgb_depth=buf_inputs[t].rgb_depth[env_i : env_i + 1],
                            grid=buf_inputs[t].grid[env_i : env_i + 1],
                            action_mask=buf_inputs[t].action_mask[env_i : env_i + 1],
                            d_theta=(
                                buf_inputs[t].d_theta[env_i : env_i + 1]
                                if buf_inputs[t].d_theta is not None
                                else None
                            ),
                        )
                        for t in range(t0, t1)
                    ]
                    chunk_inputs.append(seq_i)
                    chunk_dones_list.append(buf_dones[t0:t1, env_i])
                    chunk_actions_list.append(buf_actions[t0:t1, env_i])
                    chunk_old_lp_list.append(buf_log_probs[t0:t1, env_i])
                    chunk_adv_list.append(adv_norm[t0:t1, env_i])
                    chunk_ret_list.append(returns[t0:t1, env_i])
                    chunk_next_rays_list.append(buf_next_rays[t0:t1, env_i])
                    chunk_compass_list.append(buf_compass_tgt[t0:t1, env_i])
                    chunk_gru_list.append(chunk_init_gru[c][:, env_i : env_i + 1, :])
                    chunk_slots_list.append(chunk_init_slots[c][env_i : env_i + 1])

            model.train()
            mb_size = max(1, min(total_chunks, minibatch_chunks))
            bc_beta = max(0.0, bc_kl_beta_init * (1.0 - float(global_step) / float(bc_kl_decay_steps)))

            for _ in range(ppo_epochs):
                perm = np_rng.permutation(total_chunks)
                for start in range(0, total_chunks, mb_size):
                    mb_idx = perm[start : start + mb_size]
                    bsz_mb = len(mb_idx)

                    # Stack chunk batch -> (bsz_mb, bptt_len, ...)
                    mb_dict: dict[str, torch.Tensor] = {}
                    keys = ("rays", "ent_feats", "ent_types", "ent_mask", "vitals", "rgb_depth", "grid", "action_mask")
                    for k in keys:
                        mb_dict[k] = torch.stack(
                            [torch.cat([getattr(chunk_inputs[idx][t], k) for t in range(bptt_len)], dim=0) for idx in mb_idx],
                            dim=0,
                        ).to(device)
                    mb_dones = torch.stack([chunk_dones_list[idx] for idx in mb_idx], dim=0).to(device)
                    mb_actions = torch.stack([chunk_actions_list[idx] for idx in mb_idx], dim=0).to(device)
                    mb_old_lp = torch.stack([chunk_old_lp_list[idx] for idx in mb_idx], dim=0).to(device)
                    mb_adv = torch.stack([chunk_adv_list[idx] for idx in mb_idx], dim=0).to(device)
                    mb_ret = torch.stack([chunk_ret_list[idx] for idx in mb_idx], dim=0).to(device)
                    mb_next_rays = torch.stack([chunk_next_rays_list[idx] for idx in mb_idx], dim=0).to(device)
                    mb_compass = torch.stack([chunk_compass_list[idx] for idx in mb_idx], dim=0).to(device)

                    mb_h = RecurrentState(
                        gru_h=torch.cat([chunk_gru_list[idx] for idx in mb_idx], dim=1).to(device),
                        obj_slots=torch.cat([chunk_slots_list[idx] for idx in mb_idx], dim=0).to(device),
                    )

                    logits, values, aux, _ = bptt_unroll(
                        model,
                        mb_dict,
                        mb_dones,
                        init_h=mb_h,
                        return_aux=True,
                        rgb_dropout_p=rgb_dropout_p,
                    )

                    new_lp, entropy = masked_bernoulli_log_prob_entropy(logits, mb_dict["action_mask"], mb_actions)
                    ratio = torch.exp((new_lp - mb_old_lp).clamp(-10.0, 10.0))
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()
                    v_loss = 0.5 * (values - mb_ret).pow(2).mean()
                    ent_loss = entropy.mean()

                    assert aux is not None
                    ray_loss = F.smooth_l1_loss(aux["next_rays"], mb_next_rays)
                    compass_loss = F.mse_loss(aux["value_compass"], mb_compass)

                    # RND predictor loss
                    rnd_in = torch.cat([mb_dict["rays"].flatten(2), mb_dict["vitals"]], dim=-1).reshape(bsz_mb * bptt_len, -1)
                    _, rnd_loss = rnd.compute_intrinsic_reward(rnd_in, update_stats=False)

                    loss = (
                        pg_loss
                        + vf_coef * v_loss
                        - ent_coef * ent_loss
                        + aux_coef * ray_loss
                        + compass_coef * compass_loss
                        + rnd_loss
                    )

                    if bc_anchor is not None and bc_beta > 0.0:
                        with torch.no_grad():
                            bc_logits, _, _, _ = bptt_unroll(
                                bc_anchor, mb_dict, mb_dones, init_h=mb_h, return_aux=False
                            )
                        loss = loss + bc_beta * _bernoulli_kl(logits, bc_logits, mb_dict["action_mask"]).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(rnd.predictor.parameters()), max_grad_norm
                    )
                    optimizer.step()
                    last_loss = float(loss.item())

            if completed_returns:
                last_mean_return = float(np.mean(completed_returns))

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": last_loss,
                        "rollout/ep_return": last_mean_return,
                        "plr/bc_kl_beta": bc_beta,
                    },
                    step=global_step,
                )

            if upd % save_interval == 0 or upd == start_update + total_updates:
                last_ckpt_path = save_checkpoint(
                    ckpt_dir / f"ckpt_step_{global_step:09d}.pt",
                    model=model,
                    model_config=model_cfg,
                    global_step=global_step,
                    update_idx=upd,
                )

    finally:
        venv.close()
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass

    return {
        "global_step": global_step,
        "last_loss": last_loss,
        "mean_return": last_mean_return,
        "checkpoint": str(last_ckpt_path) if last_ckpt_path else None,
    }


run_training = train


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train Self-Evolving Recurrent PPO DOOM agent.")
    parser.add_argument("--config", type=str, default="training/configs/ppo_curriculum.yaml")
    args = parser.parse_args(argv)
    res = train(args.config)
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
