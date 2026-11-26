"""Procedural 4-Tier DOOM Map Generator (Tiers 1-4) with BFS solvability check.

Generates random, non-hardcoded UDMF PWADs from (tier, knobs, seed) so the
agent never memorizes fixed coordinates or layouts.
"""

from __future__ import annotations

import hashlib
import random
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .udmf_builder import (
    LOCK_BLUE,
    LOCK_RED,
    LOCK_YELLOW,
    THING_AMMO_BOX,
    THING_ARMOR_GREEN,
    THING_CLIP,
    THING_DEMON,
    THING_IMP,
    THING_KEY_BLUE,
    THING_KEY_RED,
    THING_KEY_YELLOW,
    THING_MEDIKIT,
    THING_PLAYER1,
    THING_SHELLS,
    THING_SHOTGUN,
    THING_SHOTGUNGUY,
    THING_STIMPACK,
    THING_ZOMBIEMAN,
    PillarSpec,
    SectorSpec,
    ThingSpec,
    UDMFMapBuilder,
)

# Macro-grid constants (Doom map units)
CELL_STEP = 384
ROOM_PAD = 32  # 2 * ROOM_PAD = 64-unit doorway/corridor depth
DOOR_WIDTH = 96
EXIT_PAD_SIZE = 64

FLOOR_TEXTURES = ("FLOOR0_1", "FLOOR4_8", "FLOOR5_1", "FLAT1", "FLAT5_4", "MFLR8_1")
CEIL_TEXTURES = ("CEIL1_1", "CEIL3_5", "FLAT1_1", "FLAT5_5", "CEIL5_1")
WALL_TEXTURES = ("STARTAN2", "STONE2", "STARGR1", "BROWN1", "GRAY1", "TEKGREN2")

KEY_THING_BY_LOCK = {
    LOCK_RED: THING_KEY_RED,
    LOCK_BLUE: THING_KEY_BLUE,
    LOCK_YELLOW: THING_KEY_YELLOW,
}


@dataclass
class _GridRoom:
    gx: int
    gy: int
    floor_h: int
    ceil_h: int
    light: int
    tex_floor: str
    tex_ceil: str
    tex_wall: str
    key_lock: int = 0  # non-zero if this room holds a keycard


@dataclass
class _GridEdge:
    a: tuple[int, int]
    b: tuple[int, int]
    is_door: bool
    lock: int = 0
    pre_opened: bool = False


def _verify_solvable(
    spawn: tuple[int, int],
    exit_node: tuple[int, int],
    rooms: dict[tuple[int, int], _GridRoom],
    edges: list[_GridEdge],
) -> bool:
    """BFS solvability check over the room-key-lock graph (generation-time only)."""
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {n: [] for n in rooms}
    for e in edges:
        adj[e.a].append((e.b, 0 if e.pre_opened else e.lock))
        adj[e.b].append((e.a, 0 if e.pre_opened else e.lock))

    held_keys: set[int] = set()
    while True:
        visited: set[tuple[int, int]] = {spawn}
        q: deque[tuple[int, int]] = deque([spawn])
        new_key = False
        while q:
            cur = q.popleft()
            if cur == exit_node:
                return True
            k = rooms[cur].key_lock
            if k and k not in held_keys:
                held_keys.add(k)
                new_key = True
            for nxt, req_lock in adj[cur]:
                if nxt not in visited and (req_lock == 0 or req_lock in held_keys):
                    visited.add(nxt)
                    q.append(nxt)
        if not new_key:
            return False


def _room_bounds(gx: int, gy: int) -> tuple[float, float, float, float]:
    x1 = gx * CELL_STEP + ROOM_PAD
    y1 = gy * CELL_STEP + ROOM_PAD
    x2 = (gx + 1) * CELL_STEP - ROOM_PAD
    y2 = (gy + 1) * CELL_STEP - ROOM_PAD
    return float(x1), float(y1), float(x2), float(y2)


class MapGen:
    """Procedural 4-tier map generator."""

    @staticmethod
    def sample(
        tier: int,
        knobs: dict[str, Any] | None = None,
        seed: int | None = None,
        out_dir: str | Path | None = None,
        scaffolding_alpha: float = 0.0,
    ) -> Path:
        """Generate a solvable UDMF PWAD for the given tier, knobs, and seed."""
        if tier not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported tier {tier}; expected 1..4")
        rng = random.Random(seed)
        knobs = dict(knobs or {})
        alpha = max(0.0, min(1.0, float(scaffolding_alpha)))

        target_dir = Path(out_dir) if out_dir is not None else Path(tempfile.gettempdir()) / "yugaantar_maps"
        target_dir.mkdir(parents=True, exist_ok=True)
        seed_tag = seed if seed is not None else rng.randint(100_000, 999_999)
        out_path = target_dir / f"proc_t{tier}_s{seed_tag}.wad"

        builder = UDMFMapBuilder("MAP01")

        # Build room graph on macro-grid
        rooms, edges, spawn_node, exit_node = _build_tier_graph(tier, knobs, rng, alpha)

        if not _verify_solvable(spawn_node, exit_node, rooms, edges):
            raise RuntimeError(f"Generated unsolvable map for tier={tier}, seed={seed_tag}")

        # Emit room sectors
        for node, r in rooms.items():
            x1, y1, x2, y2 = _room_bounds(*node)
            pillars: list[PillarSpec] = []
            if tier in (1, 4) and rng.random() < 0.75:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if node != spawn_node or rng.random() < 0.5:
                    # Place 1-2 off-center pillars so spawn center stays clear
                    px = cx + rng.choice([-72.0, 72.0])
                    py = cy + rng.choice([-72.0, 72.0])
                    pillars.append(PillarSpec(px - 24, py - 24, px + 24, py + 24, texture=r.tex_wall))
            builder.add_sector(
                SectorSpec(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    floor_h=r.floor_h,
                    ceil_h=r.ceil_h,
                    light=r.light,
                    tex_floor=r.tex_floor,
                    tex_ceil=r.tex_ceil,
                    tex_wall=r.tex_wall,
                    pillars=pillars,
                )
            )

        # Emit connecting doorway / corridor sectors
        for e in edges:
            (ax, ay), (bx, by) = e.a, e.b
            ra, rb = rooms[e.a], rooms[e.b]
            base_floor = max(ra.floor_h, rb.floor_h)
            base_ceil = min(ra.ceil_h, rb.ceil_h)
            half_w = DOOR_WIDTH / 2.0
            if bx != ax:
                # Horizontal neighbor
                mid_x = max(ax, bx) * CELL_STEP
                cy = ay * CELL_STEP + CELL_STEP / 2.0
                x1, x2 = float(mid_x - ROOM_PAD), float(mid_x + ROOM_PAD)
                y1, y2 = float(cy - half_w), float(cy + half_w)
            else:
                # Vertical neighbor
                mid_y = max(ay, by) * CELL_STEP
                cx = ax * CELL_STEP + CELL_STEP / 2.0
                x1, x2 = float(cx - half_w), float(cx + half_w)
                y1, y2 = float(mid_y - ROOM_PAD), float(mid_y + ROOM_PAD)

            builder.add_sector(
                SectorSpec(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    floor_h=base_floor,
                    ceil_h=base_ceil,
                    light=min(ra.light, rb.light),
                    tex_floor=ra.tex_floor,
                    tex_ceil=ra.tex_ceil,
                    tex_wall=ra.tex_wall,
                    is_door=e.is_door,
                    door_lock=e.lock,
                    pre_opened=e.pre_opened,
                )
            )

        # Emit Exit Alcove attached to an unused outer wall of exit_node
        ex, ey = exit_node
        rx1, ry1, rx2, ry2 = _room_bounds(ex, ey)
        er = rooms[exit_node]
        free_dirs = [d for d in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (ex + d[0], ey + d[1]) not in rooms]
        if not free_dirs:
            # Fallback: pick any room that has a free outer neighbor
            for cand_node in reversed(list(rooms.keys())):
                cx_n, cy_n = cand_node
                c_free = [d for d in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (cx_n + d[0], cy_n + d[1]) not in rooms]
                if c_free:
                    ex, ey = cand_node
                    rx1, ry1, rx2, ry2 = _room_bounds(ex, ey)
                    er = rooms[cand_node]
                    free_dirs = c_free
                    break
        dx, dy = rng.choice(free_dirs) if free_dirs else (1, 0)
        rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
        half_p = EXIT_PAD_SIZE / 2.0
        if dx == 1:
            ax1, ay1, ax2, ay2 = rx2, rcy - half_p, rx2 + EXIT_PAD_SIZE, rcy + half_p
        elif dx == -1:
            ax1, ay1, ax2, ay2 = rx1 - EXIT_PAD_SIZE, rcy - half_p, rx1, rcy + half_p
        elif dy == 1:
            ax1, ay1, ax2, ay2 = rcx - half_p, ry2, rcx + half_p, ry2 + EXIT_PAD_SIZE
        else:
            ax1, ay1, ax2, ay2 = rcx - half_p, ry1 - EXIT_PAD_SIZE, rcx + half_p, ry1
        builder.add_sector(
            SectorSpec(
                x1=ax1,
                y1=ay1,
                x2=ax2,
                y2=ay2,
                floor_h=er.floor_h,
                ceil_h=er.ceil_h,
                light=224,
                tex_floor="GATE3",
                tex_ceil=er.tex_ceil,
                tex_wall="SW1COMP",
                is_exit=True,
            )
        )

        # Populate player, keys, enemies, and pickups with randomized positions
        _populate_things(builder, tier, knobs, rng, rooms, spawn_node, exit_node, alpha)

        return builder.write_wad(out_path)


def _make_room(gx: int, gy: int, rng: random.Random, base_floor: int = 0) -> _GridRoom:
    # Safe step height <= 16 units
    floor_h = base_floor + rng.choice([0, 0, 8, 16])
    ceil_h = floor_h + rng.choice([128, 144, 160])
    return _GridRoom(
        gx=gx,
        gy=gy,
        floor_h=floor_h,
        ceil_h=ceil_h,
        light=rng.randint(128, 224),
        tex_floor=rng.choice(FLOOR_TEXTURES),
        tex_ceil=rng.choice(CEIL_TEXTURES),
        tex_wall=rng.choice(WALL_TEXTURES),
    )


def _build_tier_graph(
    tier: int,
    knobs: dict[str, Any],
    rng: random.Random,
    alpha: float,
) -> tuple[dict[tuple[int, int], _GridRoom], list[_GridEdge], tuple[int, int], tuple[int, int]]:
    rooms: dict[tuple[int, int], _GridRoom] = {}
    edges: list[_GridEdge] = []
    spawn = (0, 0)
    rooms[spawn] = _make_room(0, 0, rng)

    if tier == 1:
        num_rooms = int(knobs.get("num_rooms", rng.choice([1, 2])))
        exit_node = spawn
        if num_rooms >= 2:
            dx, dy = rng.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
            nxt = (dx, dy)
            rooms[nxt] = _make_room(dx, dy, rng)
            edges.append(_GridEdge(spawn, nxt, is_door=False))
            exit_node = nxt
        return rooms, edges, spawn, exit_node

    if tier == 2:
        num_rooms = int(knobs.get("num_rooms", rng.randint(3, 5)))
        exit_node = _grow_random_tree(rooms, edges, spawn, num_rooms, rng, door_prob=0.85, alpha=alpha)
        return rooms, edges, spawn, exit_node

    if tier == 3:
        # Tier 3: Spawn -> Branch A (keycard) & Branch B (locked door -> exit)
        lock_color = rng.choice([LOCK_BLUE, LOCK_YELLOW, LOCK_RED])
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(dirs)
        d_a, d_b = dirs[0], dirs[1]

        # Branch A: 1-2 unlocked rooms leading to keycard
        a1 = d_a
        rooms[a1] = _make_room(*a1, rng)
        pre_a = rng.random() < (alpha * 0.5)
        edges.append(_GridEdge(spawn, a1, is_door=True, lock=0, pre_opened=pre_a))
        key_node = a1
        if rng.random() < 0.6:
            a2 = _pick_free_neighbor(a1, rooms, rng)
            if a2 is not None:
                rooms[a2] = _make_room(*a2, rng)
                edges.append(_GridEdge(a1, a2, is_door=rng.random() < 0.5, lock=0, pre_opened=pre_a))
                key_node = a2
        rooms[key_node].key_lock = lock_color

        # Branch B: locked door -> exit room
        b1 = d_b
        rooms[b1] = _make_room(*b1, rng)
        pre_b = rng.random() < (alpha * 0.35)
        edges.append(_GridEdge(spawn, b1, is_door=True, lock=lock_color, pre_opened=pre_b))
        exit_node = b1
        if rng.random() < 0.4:
            b2 = _pick_free_neighbor(b1, rooms, rng)
            if b2 is not None:
                rooms[b2] = _make_room(*b2, rng)
                edges.append(_GridEdge(b1, b2, is_door=False))
                exit_node = b2
        return rooms, edges, spawn, exit_node

    # Tier 4: Full Gauntlet (5-8 rooms, 1-2 sequential key locks)
    num_rooms = int(knobs.get("num_rooms", rng.randint(5, 8)))
    num_keys = int(knobs.get("num_keys", rng.choice([1, 2])))
    locks = rng.sample([LOCK_BLUE, LOCK_YELLOW, LOCK_RED], k=min(num_keys, 2))

    # Build main spine + side branches
    cur = spawn
    lock_idx = 0
    for step in range(num_rooms - 1):
        nxt = _pick_free_neighbor(cur, rooms, rng)
        if nxt is None:
            # Backtrack to any room with a free neighbor
            candidates = [n for n in rooms if _pick_free_neighbor(n, rooms, rng) is not None]
            if not candidates:
                break
            cur = rng.choice(candidates)
            nxt = _pick_free_neighbor(cur, rooms, rng)
            if nxt is None:
                break
        rooms[nxt] = _make_room(*nxt, rng)
        # Place key in cur or a side branch before adding a locked edge
        if lock_idx < len(locks) and step in (1, 3):
            # Side branch for keycard
            key_room = _pick_free_neighbor(cur, rooms, rng)
            if key_room is not None and key_room != nxt:
                rooms[key_room] = _make_room(*key_room, rng)
                edges.append(_GridEdge(cur, key_room, is_door=True, lock=0, pre_opened=rng.random() < alpha * 0.3))
                rooms[key_room].key_lock = locks[lock_idx]
            else:
                rooms[cur].key_lock = locks[lock_idx]
            pre_lock = rng.random() < (alpha * 0.25)
            edges.append(_GridEdge(cur, nxt, is_door=True, lock=locks[lock_idx], pre_opened=pre_lock))
            lock_idx += 1
        else:
            edges.append(_GridEdge(cur, nxt, is_door=rng.random() < 0.7, lock=0, pre_opened=rng.random() < alpha * 0.4))
        cur = nxt

    return rooms, edges, spawn, cur


def _pick_free_neighbor(node: tuple[int, int], rooms: dict[tuple[int, int], _GridRoom], rng: random.Random) -> tuple[int, int] | None:
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rng.shuffle(dirs)
    for dx, dy in dirs:
        cand = (node[0] + dx, node[1] + dy)
        if cand not in rooms:
            return cand
    return None


def _grow_random_tree(
    rooms: dict[tuple[int, int], _GridRoom],
    edges: list[_GridEdge],
    spawn: tuple[int, int],
    target_rooms: int,
    rng: random.Random,
    door_prob: float,
    alpha: float,
) -> tuple[int, int]:
    cur = spawn
    farthest = spawn
    while len(rooms) < target_rooms:
        nxt = _pick_free_neighbor(cur, rooms, rng)
        if nxt is None:
            candidates = [n for n in rooms if _pick_free_neighbor(n, rooms, rng) is not None]
            if not candidates:
                break
            cur = rng.choice(candidates)
            continue
        rooms[nxt] = _make_room(*nxt, rng)
        is_door = rng.random() < door_prob
        pre = is_door and (rng.random() < alpha * 0.5)
        edges.append(_GridEdge(cur, nxt, is_door=is_door, lock=0, pre_opened=pre))
        cur = nxt
        if abs(nxt[0] - spawn[0]) + abs(nxt[1] - spawn[1]) >= abs(farthest[0] - spawn[0]) + abs(farthest[1] - spawn[1]):
            farthest = nxt
    return farthest


def _populate_things(
    builder: UDMFMapBuilder,
    tier: int,
    knobs: dict[str, Any],
    rng: random.Random,
    rooms: dict[tuple[int, int], _GridRoom],
    spawn_node: tuple[int, int],
    exit_node: tuple[int, int],
    alpha: float,
) -> None:
    enemies_per_room = int(knobs.get("enemies_per_room", {1: 3, 2: 2, 3: 2, 4: 3}[tier]))
    ammo_mult = float(knobs.get("ammo_mult", 1.0)) + alpha * 0.5

    for node, r in rooms.items():
        x1, y1, x2, y2 = _room_bounds(*node)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if node == spawn_node:
            # Randomize player spawn offset & facing angle (never fixed XYZ)
            px = cx + rng.uniform(-64.0, 64.0)
            py = cy + rng.uniform(-64.0, 64.0)
            builder.add_thing(ThingSpec(px, py, rng.choice([0, 45, 90, 135, 180, 225, 270, 315]), THING_PLAYER1))
            if rng.random() < 0.5 * ammo_mult:
                builder.add_thing(ThingSpec(cx + rng.uniform(-96, 96), cy + rng.uniform(-96, 96), 0, THING_CLIP))
            if tier >= 3 and rng.random() < 0.7:
                builder.add_thing(ThingSpec(cx - 80, cy + 80, 0, THING_SHOTGUN))

        if r.key_lock > 0:
            kx = cx + rng.uniform(-48.0, 48.0)
            ky = cy + rng.uniform(-48.0, 48.0)
            builder.add_thing(ThingSpec(kx, ky, 0, KEY_THING_BY_LOCK[r.key_lock]))

        # Enemies
        n_enemies = enemies_per_room if node != spawn_node else (2 if tier == 1 else 0)
        pool = [THING_ZOMBIEMAN, THING_SHOTGUNGUY] if tier <= 2 else [THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP]
        if tier == 4:
            pool.append(THING_DEMON)
        for _ in range(n_enemies):
            ex = cx + rng.uniform(-110.0, 110.0)
            ey = cy + rng.uniform(-110.0, 110.0)
            builder.add_thing(
                ThingSpec(
                    ex,
                    ey,
                    rng.randint(0, 359),
                    rng.choice(pool),
                    ambush=(tier == 4 and rng.random() < 0.5),
                )
            )

        # Health / Armor / Ammo pickups
        if rng.random() < 0.65 * ammo_mult:
            item = rng.choice([THING_STIMPACK, THING_CLIP, THING_SHELLS, THING_MEDIKIT, THING_ARMOR_GREEN, THING_AMMO_BOX])
            ix = cx + rng.uniform(-110.0, 110.0)
            iy = cy + rng.uniform(-110.0, 110.0)
            builder.add_thing(ThingSpec(ix, iy, 0, item))


def wad_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
