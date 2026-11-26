"""Live split-screen DOOM + Raycast/Entity/Depth visualizer (pygame).

Left panel:  RGB game view (obs["rgb"])
Right panel: Live 360° raycast radar + 1D depth/hit-type view + visible_entities + vitals HUD

Works on any procedural Tier 1-4 map or WAD using only the official `doom_env` observation dict.

Controls:
  W/S or Up/Down : Move forward / backward
  A/D            : Strafe left / right
  Left/Right     : Turn left / right
  Space / F      : Attack (shoot)
  E              : Use (open door / press switch)
  Q / Esc        : Quit
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pygame

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "template_reference" / "env"))

from doom_env import make_env  # noqa: E402
from training.mapgen.tiers import MapGen  # noqa: E402

# Hit-type colors matching the video's debug palette
HIT_COLORS = {
    0: (40, 40, 48),       # none
    1: (220, 220, 230),    # wall (white/gray)
    2: (60, 160, 255),     # door (blue)
    3: (255, 200, 40),     # locked door (gold)
    4: (255, 60, 60),      # enemy (red)
    5: (60, 230, 110),     # item (green)
}

HIT_NAMES = {0: "None", 1: "Wall", 2: "Door", 3: "Locked Door", 4: "Enemy", 5: "Item"}


def _read_keyboard_action() -> np.ndarray:
    keys = pygame.key.get_pressed()
    act = np.zeros(9, dtype=np.int8)
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        act[0] = 1  # MOVE_FORWARD
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        act[1] = 1  # MOVE_BACKWARD
    if keys[pygame.K_a]:
        act[2] = 1  # MOVE_LEFT
    if keys[pygame.K_d]:
        act[3] = 1  # MOVE_RIGHT
    if keys[pygame.K_LEFT]:
        act[4] = 1  # TURN_LEFT
    if keys[pygame.K_RIGHT]:
        act[5] = 1  # TURN_RIGHT
    if keys[pygame.K_SPACE] or keys[pygame.K_f]:
        act[6] = 1  # ATTACK
    if keys[pygame.K_e]:
        act[7] = 1  # USE
    if keys[pygame.K_TAB]:
        act[8] = 1  # SELECT_NEXT_WEAPON
    return act


def _draw_right_panel(
    screen: pygame.Surface,
    obs: dict,
    font: pygame.font.Font,
    x0: int,
    w: int,
    h: int,
) -> None:
    pygame.draw.rect(screen, (14, 16, 22), (x0, 0, w, h))

    # 1. Top half: Pseudo-3D raycast columns (mirrors what the agent "sees" in 90° FOV)
    fov_h = h // 2
    pygame.draw.rect(screen, (20, 22, 30), (x0 + 10, 10, w - 20, fov_h - 15))
    rays_dist = obs["rays"]["distance"]
    rays_type = obs["rays"]["hit_type"]
    # Front 90° FOV corresponds to ray indices [56..63] and [0..8] (17 rays)
    fov_indices = list(range(8, -1, -1)) + list(range(63, 55, -1))
    col_w = (w - 24) / len(fov_indices)
    mid_y = 10 + (fov_h - 15) // 2
    max_bar_h = fov_h - 24

    for ci, r_idx in enumerate(fov_indices):
        d = float(rays_dist[r_idx])
        ht = int(rays_type[r_idx])
        color = HIT_COLORS.get(ht, (200, 200, 200))
        # Perspective height ~ 1 / distance
        bar_h = int(min(max_bar_h, max_bar_h * (0.14 / max(0.05, d))))
        shade = max(0.25, 1.0 - 0.7 * d)
        shaded = tuple(int(c * shade) for c in color)
        rx = int(x0 + 12 + ci * col_w)
        ry = mid_y - bar_h // 2
        pygame.draw.rect(screen, shaded, (rx, ry, max(2, int(col_w) - 1), bar_h))

    # Overlay visible_entities bounding markers in FOV
    ent_feats = obs["visible_entities"]["features"]
    ent_types = obs["visible_entities"]["entity_type"]
    ent_mask = obs["visible_entities"]["mask"]
    for i in range(len(ent_mask)):
        if not ent_mask[i]:
            continue
        dist_norm, sin_a, cos_a, _, _ = ent_feats[i]
        rel_angle = math.atan2(sin_a, cos_a)  # 0 = ahead, +left, -right
        if abs(rel_angle) <= math.radians(48):
            frac_x = 0.5 - (rel_angle / math.radians(90))
            ex = int(x0 + 12 + frac_x * (w - 24))
            box_sz = int(min(max_bar_h * 0.7, max(12, 24 / max(0.08, dist_norm))))
            etype = int(ent_types[i])
            ecolor = (255, 50, 50) if etype in (1, 2, 3) else (60, 230, 110)
            pygame.draw.rect(screen, ecolor, (ex - box_sz // 2, mid_y - box_sz // 2, box_sz, box_sz), 2)

    # 2. Bottom-left: 360° Top-Down Raycast Radar
    cx = x0 + 140
    cy = fov_h + (h - fov_h) // 2 + 10
    radius = min(115, (h - fov_h) // 2 - 20)
    pygame.draw.circle(screen, (28, 32, 42), (cx, cy), radius, 1)
    pygame.draw.circle(screen, (28, 32, 42), (cx, cy), radius // 2, 1)

    for i in range(64):
        d = float(rays_dist[i])
        ht = int(rays_type[i])
        theta = (i / 64.0) * 2.0 * math.pi  # 0 = straight ahead (up on screen), CCW
        px = cx - int(math.sin(theta) * d * radius)
        py = cy - int(math.cos(theta) * d * radius)
        col = HIT_COLORS.get(ht, (180, 180, 180))
        pygame.draw.line(screen, (col[0] // 3, col[1] // 3, col[2] // 3), (cx, cy), (px, py), 1)
        if ht != 0:
            pygame.draw.circle(screen, col, (px, py), 3)

    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 4)

    # 3. Bottom-right: Vitals & Legend
    lx = x0 + 280
    ly = fov_h + 20
    v = obs["vitals"]
    stats = [
        f"Health: {v[0]:.0f}   Armor: {v[1]:.0f}",
        f"Weapon: {v[2]:.0f}   Ammo: {v[3]:.0f}",
        f"Keys (B/Y/R): {int(v[8])}/{int(v[9])}/{int(v[10])}",
        f"Vel F/R: {v[14]:.1f} / {v[15]:.1f}",
        f"Visible Entities: {int(ent_mask.sum())}",
    ]
    for s in stats:
        screen.blit(font.render(s, True, (220, 225, 235)), (lx, ly))
        ly += 22

    ly += 10
    for ht in (1, 2, 3, 4, 5):
        pygame.draw.rect(screen, HIT_COLORS[ht], (lx, ly + 3, 12, 12))
        screen.blit(font.render(HIT_NAMES[ht], True, (180, 185, 195)), (lx + 18, ly))
        ly += 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-screen DOOM + live ray/entity visualizer.")
    parser.add_argument("--tier", type=int, default=3, choices=[1, 2, 3, 4], help="Procedural map tier.")
    parser.add_argument("--seed", type=int, default=1001, help="Map seed.")
    parser.add_argument("--wad", type=str, default=None, help="Optional custom WAD path instead of MapGen.")
    args = parser.parse_args()

    wad_path = Path(args.wad) if args.wad else MapGen.sample(tier=args.tier, seed=args.seed)
    env = make_env(wad_path, "MAP01", episode_ticks=4200, render_buffers=True)

    pygame.init()
    left_w, right_w, win_h = 480, 500, 360
    screen = pygame.display.set_mode((left_w + right_w, win_h))
    pygame.display.set_caption(f"Yugaantar DOOM Live Sensor View — {wad_path.name}")
    font = pygame.font.SysFont("menlo", 14)
    clock = pygame.time.Clock()

    obs, _ = env.reset(seed=0)
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            act = _read_keyboard_action()
            obs, _, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                print(f"Episode ended (exit_reached={info.get('exit_reached')}). Resetting...")
                obs, _ = env.reset(seed=0)

            # Left: RGB buffer scaled to 480x360
            rgb_surf = pygame.surfarray.make_surface(np.transpose(obs["rgb"], (1, 0, 2)))
            rgb_scaled = pygame.transform.scale(rgb_surf, (left_w, win_h))
            screen.blit(rgb_scaled, (0, 0))

            # Right: Live 3D-ray / entity / radar / vitals visualizer
            _draw_right_panel(screen, obs, font, left_w, right_w, win_h)
            pygame.display.flip()
            clock.tick(15)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
