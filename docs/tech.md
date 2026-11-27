# DevForge "Can It Play DOOM?" — Complete Technical Guide (Easy Language)

> **Team:** Aryan, Vedesh & Team | **Commit:** Latest Championship Submission (`b8dbd2a` / `37bf78c`)
> **Best Eval Results:** **`2,934.33`** (`51.9%` exit rate — **14/27 maps**!) & **`3,448.99`** Championship Run (`44.4%` exit rate)
> **Docker Sandbox (`validate.py --docker`):** ✅ **PASSED** (`0` warnings, `72` steps to exit, `13.3 ms` p50)
> **Unit Tests (`pytest`):** ✅ **247 / 247 PASSED**

---

## 🎯 What Is This Project? (The 30-Second Pitch)

We built an AI agent that plays DOOM **completely on its own** on **brand-new 3D maps it has never seen before** — it navigates rooms, fights enemies, picks up keycards, unlocks doors, and finds the exit — **with zero map coordinates** and **zero hardcoded pathfinding**.

---

## 💡 What Makes It Special? (Our 3 Big Ideas)

### 1. Zero GPS Coordinates (100% Ego-Centric)
Most DOOM bots cheat during training or evaluation by reading world coordinates like `(x=1024, y=-512)`. Our agent gets **zero coordinates**. Instead, it builds its own mental map of where it has been using only what it can *sense* right now — like a person navigating in the dark using sonar echoes. That is why it works on brand-new maps.

### 2. 5 Senses Combined Into One Brain
The agent doesn't just stare at camera pixels. It combines:
- 🔭 **64 radar rays** shooting 360° around the player (like lidar/sonar)
- 👁️ **Camera view** (`48×64` RGB + Depth — sees colors AND distances)
- 📍 **Self-built mental map** (`6×32×32` grid it draws as it moves)
- 👾 **Enemy/item tracker** (up to 32 nearby objects in its 90° view)
- 💊 **Vitals** (health, armor, weapon, ammo, 6 keycard flags, speed, previous button pressed)

### 3. Learned From a Smart Teacher First, Then Practiced Alone
Pure trial-and-error (RL) stalled at a **3.7% exit rate** — the agent kept spinning in corners. We broke through by having a **privileged teacher bot** (used *only* during training, allowed by rules) show the student agent how to fight, open doors, and connect keycards to locked doors (**DAgger**), and then letting the student practice on its own with **BC-KL-anchored PPO** (`ppo_championship.yaml`). Exit rate jumped all the way to **51.9%**!

---

## 🧠 How Does It Think? (`HybridDoomNet` — `868,403` Parameters)

Inside [submission/model.py](file:///c:/Users/aryan/Desktop/scaler%20one/hahahaahahaha/submission/model.py), the brain has **5 sensor processors** feeding into a **memory core** (well under the `2,000,000` parameter limit):

```
[64 Radar Rays]  ──▶ [Circular 1D Conv (96)]  ─┐
[Camera RGB+D]   ──▶ [Depthwise 2D CNN (64)]   ─┤
[32×32 Ego Grid] ──▶ [2D Grid CNN (64)]        ─┼──▶ [Fusion (256)] ──▶ [GRU (256)] ──▶ [9 Actions]
[32 Entities]    ──▶ [Self-Attention (64)]     ─┤                           │
[36 Vitals]      ──▶ [MLP (48)]                ─┘                   [4×8 Goal Memory Slots]
```

### Every Small Detail Inside the Brain:
- **Circular Ray Padding (`padding_mode="circular"`):** The 64 rays form a 360° circle, so ray `63` wraps seamlessly into ray `0` behind the player — no blind seam.
- **NaN-Safe Entity Attention:** Standard PyTorch Self-Attention crashes with `NaN` if `0` enemies are visible. Our `EntitySetEncoder` explicitly guards empty masks to return clean zeros.
- **RGB Modality Dropout (`20–25%` during training):** We randomly black out the camera during training so the agent never becomes dependent on wall colors/textures.
- **GRU (`256` dims):** Short-term working memory across the whole episode (reset when a map ends).
- **4 × 8 Gated Objective Memory Slots:** Sigmoid-gated "sticky notes" that remember long-term goals (like *"there is a locked door back there, go find the key"*) across hundreds of steps.
- **9 Binary Outputs:** Move Forward, Move Backward, Strafe Left, Strafe Right, Turn Left, Turn Right, Attack, Use (open door/exit), Run.

---

## 🗺️ How Does It Build Its Own Map? ([submission/preprocessing.py](file:///c:/Users/aryan/Desktop/scaler%20one/hahahaahahaha/submission/preprocessing.py))

This is called **Scan-Matched Egocentric Occupancy Grid Mapping**:
1. Every frame, 64 radar rays shoot out in all directions (`RAY_RANGE = 1024`).
2. `_estimate_dtheta` ignores moving enemies and max-range misses, then compares **this frame's static rays** against **last frame's static rays** (using circular shifts + 3-point parabolic smoothing) to figure out how much the player turned (`Δθ`).
3. It rotates and shifts its `32×32` mental map with `cv2.warpAffine` so the map always stays centered on the player.
4. It paints what the rays hit across 6 channels: `visited`, `wall`, `door`, `locked_door`, `item`, `enemy`.
5. Old cells slowly fade (`0.998` decay for `visited` and `locked_door`, `0.992` for others) so small rotation drift never piles up.

---

## 🛡️ Rules Compliance & Evaluation Safeguards ([submission/agent.py](file:///c:/Users/aryan/Desktop/scaler%20one/hahahaahahaha/submission/agent.py))

Every action during evaluation comes from `HybridDoomNet` — zero `A*`, zero `BFS`, zero `atan2` aiming, zero `privileged_state()`, and zero `if/else` action trees.

Here are the **only** deterministic safeguards in `submission/agent.py` (all 100% legal under `docs/RULES.md`):
1. **Strict Weight Loading (`load_strict_checkpoint`):** Uses `strict=True` and fails loudly if `best_agent.pt` is missing, corrupt, mismatched, or an unfetched Git LFS pointer.
2. **1-CPU Thread Limit:** `torch.set_num_threads(1)` on CPU so PyTorch doesn't fight with ViZDoom for CPU time.
3. **Calibrated Sampling (`tau = 0.85`):** `sigmoid(logits / 0.85)` with seeded `default_rng(0)` reset per map. Breaks symmetric doorway/corner loops that trap greedy (`> 0.5`) agents.
4. **Opposite-Button Tie-Breaking:** If both Forward+Backward (or Left+Right, or Turn L+Turn R) fire together, it keeps the one with the higher neural network logit.
5. **2 Legal Observation Masks:**
   - **`USE` cooldown (6 steps):** If facing a door (`dist < 0.15`), prevents holding `USE` so the agent doesn't immediately close a door it just opened.
   - **Empty-ammo guard:** Disables `ATTACK` when `selected_weapon > 1` and `ammo <= 0`.
6. **NaN Safety:** `np.nan_to_num` clamps logits and observations.

---

## 🔢 How Was It Trained?

1. **Procedural UDMF Map Generator (`training/mapgen/`):**
   - Pure-Python generator creating 4 tiers of maps: Tier 1 (open arenas) → Tier 2 (multi-room doors) → Tier 3 (keycards + locked doors) → Tier 4 (multi-lock mazes + ambushes).
   - Randomizes origin offsets `[-2048, 2048]`, `90°` rotations, room sizes, lighting, and textures.
   - Verifies **BFS solvability** at generation time (`spawn → keycards → locked doors → exit`).
   - **Strict Train/Eval Split:** Training seeds `[100,000 – 999,999]`; held-out eval seeds `[1,000 – 9,999]` and `[50,001 – 50,016]` + 3 real `freedoom2.wad` maps.
2. **Dense Coordinate-Free Rewards (`training/rewards.py`):**
   - Rewards exits (`+15` + speed bonus), unlocking doors, keycards, opening doors, kills, new sectors/cells explored, and pickups; penalizes death, damage, wall-scraping, door-bouncing, and wasted shots.
3. **Training Progression:**
   - **Random baseline:** `0.00` score, `0.0%` exits
   - **Early RL (no curriculum):** `344–504` score, `3.7%` exits
   - **5-Stage Curriculum:** `1,440.45` score, `14.8%` exits
   - **Teacher DAgger → BC-KL PPO:** `2,126.58` score, `22.2%` exits
   - **Championship PPO (`ppo_v2` / `ppo_championship`):** **`2,934.33`** baseline (**`51.9%` exit rate**, `14/27` maps, **`6/6` Tier 1**, **`6/7` Tier 2**) and **`3,448.99`** championship score (`44.4%` exit rate)

---

## ⚡ Speed, Sandbox & Test Numbers

| Check | Result | Details |
|---|---|---|
| **Official Docker Sandbox (`validate.py --docker`)** | ✅ **PASSED (0 warnings)** | `72` steps to exit `two_rooms.wad`, `13.3 ms` p50 / `20.7 ms` p99 (limit `40 ms`), `1571 MB / 8192 MB` RAM, `~208 MB / 2048 MB` VRAM |
| **Held-Out Latency (`eval_results_baseline.json`)** | ✅ **PASSED** | `1.87 ms` mean / `4.62 ms` p99 `act()` across 27 unseen maps |
| **Full Unit Test Suite (`pytest`)** | ✅ **247 / 247 PASSED** | Covers interfaces, NaN-safety, curriculum, mapgen, audit fixes |
| **Repository Size & LFS** | ✅ **9.8 MB / 1.0 GB** | `submission/weights/best_agent.pt` tracked via Git LFS |

### Tier-by-Tier Breakdown (`eval_results_baseline.json`, 27 Unseen Maps)
- **Tier 1 (Easy):** **6/6 exits (100%)**, `100%` exploration, `0` deaths, `78.73` avg score
- **Tier 2 (Multi-room):** **6/7 exits (85.7%)**, `87.9%` exploration, `5.71` avg kills, `122.79` avg score
- **Tier 3 (Keycards & Locks):** `133.69` avg score, `5` locks unlocked, `6` keys collected
- **Tier 4 (Mazes & Ambushes):** `95.22` avg score, heavy combat survival

---

## 🙋 Judge Q&A — Plain English Answers

### Q1: "How do we know your agent isn't using `A*`, coordinates, or `if/else` tricks at evaluation?"
Look at `submission/agent.py`. `act()` runs `preprocess()` → `HybridDoomNet` → `tau=0.85` sampling → two observation-legal masks (`USE` 6-step door cooldown and empty-ammo `ATTACK` mask). Zero pathfinding, zero `atan2`, zero `privileged_state()`, and zero `if/else` action rules.

### Q2: "Isn't using a Teacher Bot during training against the rules?"
No — `docs/RULES.md §2` explicitly allows any training-time scaffolding, privileged state, or teacher scripts. `training/bc/teacher_bot.py` is **never imported or run** during evaluation.

### Q3: "What if the hidden evaluation maps look completely different?"
During training, we randomize wall textures, lighting, room sizes, origin offsets, and `90°` rotations — and we randomly turn off the camera (`20–25%` RGB dropout) so the agent relies primarily on geometry-based 64-ray radar and its self-built `32×32` map.

### Q4: "How does it remember a locked door after walking across the map for a key?"
Two systems work together: the `32×32` egocentric grid keeps `locked_door` marked with `0.998` slow decay, and the `GRU(256)` + `4×8` gated Objective Memory Slots hold persistent goal memory across hundreds of ticks.

### Q5: "Why `tau=0.85` instead of greedy `> 0.5`?"
Purely greedy thresholding gets stuck oscillating in symmetric doorways/corners. Our 27-map ablation showed `tau=0.85` breaks limit cycles while staying faithful to trained weights.

### Q6: "What is your biggest remaining limitation?"
On the hardest Tier 3–4 multi-lock mazes, the agent unlocks doors and fights well, but can still run out of time backtracking after collecting keys or take heavy early damage in Tier 4 ambushes. Extending truncated BPTT (`32 → 64`) and longer Tier 3–4 PPO fine-tuning would push those exits even higher.
