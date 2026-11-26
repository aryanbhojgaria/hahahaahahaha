"""Live split-screen DOOM + Raycast/Entity/Depth + Neural Grid visualizer (pygame).

Supports TWO modes:
1. Live Agent Spectator (recommended while training!):
     python -m training.visualize_obs --checkpoint runs/ppo_flagship/checkpoints/latest.pt --watch --random-maps
   Runs DoomAgent autonomously across random Tier 1-4 maps (or a fixed WAD),
   rendering:
     - Left panel:   First-person RGB view + live Action HUD
     - Middle panel: Pseudo-3D Raycast FOV + visible_entities boxes + 360° Polar Radar
     - Right panel:  32x32 Egocentric Occupancy Grid + Vitals + Live Training Checkpoint Stats
   Automatically hot-reloads new weights whenever the checkpoint updates!

2. Manual Keyboard Mode (no --checkpoint):
     python -m training.visualize_obs --tier 3 --seed 1001
   Controls: W/S (fwd/back), A/D (strafe), Left/Right (turn), Space/F (shoot), E (use), R (new random map), Q/Esc (quit).
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pygame

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "template_reference" / "env"))

from doom_env import make_env  # noqa: E402
from training.mapgen.tiers import MapGen  # noqa: E402
from training.odometry_grid import (  # noqa: E402
    CH_DOOR,
    CH_ENEMY,
    CH_ITEM,
    CH_LOCKED_DOOR,
    CH_VISITED,
    CH_WALL,
)

# Hit-type colors matching the live debug palette (RGB for pygame)
HIT_COLORS = {
    0: (35, 38, 46),       # none
    1: (220, 220, 230),    # wall (white/gray)
    2: (60, 160, 255),     # door (blue)
    3: (255, 200, 40),     # locked door (gold)
    4: (255, 60, 60),      # enemy (red)
    5: (60, 230, 110),     # item (green)
}

HIT_NAMES = {0: "None", 1: "Wall", 2: "Door", 3: "Locked", 4: "Enemy", 5: "Item"}
ACT_LABELS = ("FWD", "BAK", "L", "R", "TL", "TR", "ATK", "USE", "WPN")


def _read_keyboard_action() -> np.ndarray:
    keys = pygame.key.get_pressed()
    act = np.zeros(9, dtype=np.int8)
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        act[0] = 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        act[1] = 1
    if keys[pygame.K_a]:
        act[2] = 1
    if keys[pygame.K_d]:
        act[3] = 1
    if keys[pygame.K_LEFT]:
        act[4] = 1
    if keys[pygame.K_RIGHT]:
        act[5] = 1
    if keys[pygame.K_SPACE] or keys[pygame.K_f]:
        act[6] = 1
    if keys[pygame.K_e]:
        act[7] = 1
    if keys[pygame.K_TAB]:
        act[8] = 1
    return act


def _draw_mid_panel(
    screen: pygame.Surface,
    obs: dict[str, Any],
    font: pygame.font.Font,
    x0: int,
    w: int,
    h: int,
) -> None:
    pygame.draw.rect(screen, (14, 16, 22), (x0, 0, w, h))

    # 1. Top half: Pseudo-3D raycast columns + visible_entities bounding boxes
    fov_h = h // 2
    pygame.draw.rect(screen, (20, 22, 30), (x0 + 8, 8, w - 16, fov_h - 14))
    rays_dist = obs["rays"]["distance"]
    rays_type = obs["rays"]["hit_type"]
    fov_indices = list(range(8, -1, -1)) + list(range(63, 55, -1))
    col_w = (w - 20) / len(fov_indices)
    mid_y = 8 + (fov_h - 14) // 2
    max_bar_h = fov_h - 24

    for ci, r_idx in enumerate(fov_indices):
        d = float(rays_dist[r_idx])
        ht = int(rays_type[r_idx])
        color = HIT_COLORS.get(ht, (200, 200, 200))
        bar_h = int(min(max_bar_h, max_bar_h * (0.14 / max(0.05, d))))
        shade = max(0.25, 1.0 - 0.7 * d)
        shaded = tuple(int(c * shade) for c in color)
        rx = int(x0 + 10 + ci * col_w)
        ry = mid_y - bar_h // 2
        pygame.draw.rect(screen, shaded, (rx, ry, max(2, int(col_w) - 1), bar_h))

    ent_feats = obs["visible_entities"]["features"]
    ent_types = obs["visible_entities"]["entity_type"]
    ent_mask = obs["visible_entities"]["mask"]
    for i in range(len(ent_mask)):
        if not ent_mask[i]:
            continue
        dist_norm, sin_a, cos_a, _, _ = ent_feats[i]
        rel_angle = math.atan2(sin_a, cos_a)
        if abs(rel_angle) <= math.radians(48):
            frac_x = 0.5 - (rel_angle / math.radians(90))
            ex = int(x0 + 10 + frac_x * (w - 20))
            box_sz = int(min(max_bar_h * 0.7, max(12, 24 / max(0.08, dist_norm))))
            etype = int(ent_types[i])
            ecolor = (255, 60, 60) if etype in (1, 2, 3, 4) else (60, 230, 110)
            pygame.draw.rect(screen, ecolor, (ex - box_sz // 2, mid_y - box_sz // 2, box_sz, box_sz), 2)

    screen.blit(font.render("90deg Depth/Raycast + Entities", True, (180, 190, 210)), (x0 + 14, 12))

    # 2. Bottom half: 360° Top-Down Raycast Radar
    cx = x0 + w // 2
    cy = fov_h + (h - fov_h) // 2 + 4
    radius = min(w // 2 - 20, (h - fov_h) // 2 - 18)
    pygame.draw.circle(screen, (32, 36, 48), (cx, cy), radius, 1)
    pygame.draw.circle(screen, (28, 32, 42), (cx, cy), radius // 2, 1)

    for i in range(64):
        d = float(rays_dist[i])
        ht = int(rays_type[i])
        theta = (i / 64.0) * 2.0 * math.pi
        px = cx - int(math.sin(theta) * d * radius)
        py = cy - int(math.cos(theta) * d * radius)
        col = HIT_COLORS.get(ht, (180, 180, 180))
        pygame.draw.line(screen, (col[0] // 3, col[1] // 3, col[2] // 3), (cx, cy), (px, py), 1)
        if ht != 0:
            pygame.draw.circle(screen, col, (px, py), 3)

    pygame.draw.circle(screen, (255, 255, 0), (cx, cy), 4)
    screen.blit(font.render("360deg Radar", True, (180, 190, 210)), (x0 + 14, fov_h + 6))


def _draw_right_panel(
    screen: pygame.Surface,
    obs: dict[str, Any],
    act: np.ndarray,
    grid: Optional[np.ndarray],
    hud_meta: dict[str, Any],
    font: pygame.font.Font,
    x0: int,
    w: int,
    h: int,
) -> None:
    pygame.draw.rect(screen, (18, 20, 28), (x0, 0, w, h))

    # 1. Top: 32x32 Egocentric Neural Grid (rendered as crisp pixels)
    grid_box = min(w - 24, h // 2 - 12)
    gx0 = x0 + (w - grid_box) // 2
    gy0 = 24
    if grid is not None and grid.shape == (6, 32, 32):
        comp = np.zeros((32, 32, 3), dtype=np.float32)
        comp[:, :, 2] += grid[CH_VISITED] * 0.45      # blue visited
        comp[:, :, :] += grid[CH_WALL][..., None] * 0.65
        comp[:, :, 0] += grid[CH_DOOR] * 0.90
        comp[:, :, 1] += grid[CH_DOOR] * 0.90
        comp[:, :, 0] += grid[CH_LOCKED_DOOR] * 1.00
        comp[:, :, 1] += grid[CH_ITEM] * 0.85
        comp[:, :, 2] += grid[CH_ITEM] * 0.90
        comp[:, :, 1] += grid[CH_ENEMY] * 1.00
        img_u8 = (np.clip(comp, 0.0, 1.0) * 255.0).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.transpose(img_u8, (1, 0, 2)))
        scaled = pygame.transform.scale(surf, (grid_box, grid_box))
        screen.blit(scaled, (gx0, gy0))
        pygame.draw.rect(screen, (60, 70, 90), (gx0, gy0, grid_box, grid_box), 1)
        pcx, pcy = gx0 + grid_box // 2, gy0 + grid_box // 2
        pygame.draw.circle(screen, (255, 255, 0), (pcx, pcy), 4)
        pygame.draw.line(screen, (255, 255, 0), (pcx, pcy), (pcx, pcy - 10), 2)
    else:
        pygame.draw.rect(screen, (26, 28, 36), (gx0, gy0, grid_box, grid_box))

    screen.blit(font.render("32x32 Egocentric Grid", True, (180, 190, 210)), (x0 + 12, 4))

    # 2. Bottom: Vitals, Active Actions & Checkpoint Telemetry
    ly = gy0 + grid_box + 10
    v = obs["vitals"]
    lines = [
        f"Mode: {hud_meta.get('mode', 'MANUAL')}",
        f"Map:  {hud_meta.get('map_label', 'MAP01')}",
        f"Train Step: {hud_meta.get('ckpt_step', 0):,} (upd {hud_meta.get('ckpt_upd', 0)})",
        f"Ep #{hud_meta.get('episode', 1)} | Step {hud_meta.get('ep_step', 0)} | Exits {hud_meta.get('exits', 0)}",
        f"HP {v[0]:.0f} | Arm {v[1]:.0f} | Ammo {v[3]:.0f} | Key {int(v[8]+v[9]+v[10])}",
    ]
    for s in lines:
        screen.blit(font.render(s, True, (220, 225, 235)), (x0 + 12, ly))
        ly += 19

    # Action pills row
    ly += 4
    ax = x0 + 12
    for i, lbl in enumerate(ACT_LABELS):
        active = int(act[i]) == 1
        bg = (40, 180, 90) if active else (38, 42, 54)
        fg = (10, 15, 10) if active else (130, 138, 155)
        bw = 28
        pygame.draw.rect(screen, bg, (ax, ly, bw, 18), border_radius=3)
        screen.blit(font.render(lbl[:3], True, fg), (ax + 3, ly + 2))
        ax += bw + 3


def _load_agent_weights(agent: Any, ckpt_path: Path) -> tuple[int, int]:
    import torch

    ckpt = torch.load(ckpt_path, map_location=agent.device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    agent.model.load_state_dict(state_dict, strict=False)
    agent.model.eval()
    return int(ckpt.get("global_step", 0)), int(ckpt.get("update_idx", 0))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Split-screen DOOM live sensor & training spectator.")
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2, 3, 4], help="Procedural map tier.")
    parser.add_argument("--seed", type=int, default=1001, help="Initial map seed.")
    parser.add_argument("--wad", type=str, default=None, help="Optional fixed WAD path instead of MapGen.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint to drive DoomAgent.")
    parser.add_argument("--watch", action="store_true", help="Auto-reload checkpoint when modified during training.")
    parser.add_argument("--random-maps", action="store_true", help="Sample fresh Tier 1-4 maps on each episode.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fps", type=int, default=15, help="Spectator render FPS.")
    args = parser.parse_args(argv)

    tmp_wad_dir = Path(tempfile.mkdtemp(prefix="doom-vis-"))
    rng = np.random.default_rng(int(time.time()) % 100_000)

    def _next_wad(ep_idx: int) -> tuple[Path, str]:
        if args.wad:
            return Path(args.wad), Path(args.wad).name
        if args.random_maps:
            tier = int(rng.choice([1, 2, 2, 3, 4]))
            seed = int(rng.integers(200_000, 999_999))
        else:
            tier = args.tier
            seed = args.seed + ep_idx - 1
        wad_p, _ = MapGen.sample(tier=tier, knobs={"out_dir": str(tmp_wad_dir)}, seed=seed)
        return Path(wad_p), f"T{tier} s={seed}"

    agent = None
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    last_mtime = -1.0
    ckpt_step, ckpt_upd = 0, 0
    if ckpt_path is not None:
        from gymnasium.spaces import Box, MultiBinary
        from submission.agent import DoomAgent

        agent = DoomAgent(Box(-np.inf, np.inf, (1,), np.float32), MultiBinary(9), device=args.device)
        if ckpt_path.is_file():
            try:
                ckpt_step, ckpt_upd = _load_agent_weights(agent, ckpt_path)
                last_mtime = ckpt_path.stat().st_mtime
                print(f"[Spectator] Loaded {ckpt_path} (step={ckpt_step:,}, upd={ckpt_upd})", flush=True)
            except Exception as exc:
                print(f"[Spectator] Waiting for readable checkpoint ({exc})...", flush=True)

    ep_idx = 1
    exits_count = 0
    ep_step = 0
    wad_path, map_label = _next_wad(ep_idx)
    env = make_env(wad_path, "MAP01", episode_ticks=2100, render_buffers=True)

    pygame.init()
    left_w, mid_w, right_w, win_h = 480, 360, 290, 380
    screen = pygame.display.set_mode((left_w + mid_w + right_w, win_h))
    pygame.display.set_caption(f"DevForge DOOM Live View — {map_label}")
    font = pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", 13)
    clock = pygame.time.Clock()

    obs, _ = env.reset(seed=0)
    if agent is not None:
        agent.reset()

    last_watch_check = time.perf_counter()
    running = True
    try:
        while running:
            force_new_map = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        force_new_map = True

            # Hot-reload checkpoint every ~2 seconds if --watch is active
            if agent is not None and args.watch and ckpt_path is not None:
                now = time.perf_counter()
                if now - last_watch_check >= 2.0:
                    last_watch_check = now
                    if ckpt_path.is_file():
                        try:
                            mt = ckpt_path.stat().st_mtime
                            if mt > last_mtime:
                                ckpt_step, ckpt_upd = _load_agent_weights(agent, ckpt_path)
                                last_mtime = mt
                                print(f"[Spectator] Hot-reloaded weights -> step={ckpt_step:,} upd={ckpt_upd}", flush=True)
                        except Exception:
                            pass

            act = agent.act(obs) if agent is not None else _read_keyboard_action()
            obs, _, terminated, truncated, info = env.step(act)
            ep_step += 1

            if terminated or truncated or force_new_map:
                if info.get("exit_reached", False):
                    exits_count += 1
                ep_idx += 1
                ep_step = 0
                if args.random_maps or force_new_map:
                    env.close()
                    wad_path, map_label = _next_wad(ep_idx)
                    env = make_env(wad_path, "MAP01", episode_ticks=2100, render_buffers=True)
                    pygame.display.set_caption(f"DevForge DOOM Live View — {map_label}")
                obs, _ = env.reset(seed=ep_idx)
                if agent is not None:
                    agent.reset()

            rgb_surf = pygame.surfarray.make_surface(np.transpose(obs["rgb"], (1, 0, 2)))
            rgb_scaled = pygame.transform.scale(rgb_surf, (left_w, win_h))
            screen.blit(rgb_scaled, (0, 0))

            _draw_mid_panel(screen, obs, font, left_w, mid_w, win_h)
            grid = agent.odom_state.grid if (agent is not None and agent.odom_state is not None) else None
            hud_meta = {
                "mode": "AGENT LIVE (WATCH)" if (agent and args.watch) else ("AGENT LIVE" if agent else "MANUAL"),
                "map_label": map_label,
                "ckpt_step": ckpt_step,
                "ckpt_upd": ckpt_upd,
                "episode": ep_idx,
                "ep_step": ep_step,
                "exits": exits_count,
            }
            _draw_right_panel(screen, obs, act, grid, hud_meta, font, left_w + mid_w, right_w, win_h)
            pygame.display.flip()
            clock.tick(max(1, args.fps))
    finally:
        env.close()
        pygame.quit()
        import shutil
        shutil.rmtree(tmp_wad_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
