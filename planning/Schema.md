# Data Contracts, Tensor Schemas & Config Specifications

> **Freeze Rule:** All shapes, keys, and file formats in this document freeze at **H0.5**. Vedesh, Aryan, Aman, and Swapnil code independently against these exact schemas.

---

## 1. Official `doom_env` Contract (`env/doom_env/spaces.py`)

### 1.1 Raw Observation Dictionary (`obs: dict`)
| Key | Sub-key | Shape | Dtype | Range / Values | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rgb` | — | `(120, 160, 3)` | `uint8` | `[0, 255]` | HWC RGB24, HUD/crosshair hidden (`zeros` if `render_buffers=False`). |
| `depth` | — | `(120, 160)` | `uint8` | `[0, 255]` | HW aligned with `rgb` (`zeros` if `render_buffers=False`). |
| `vitals` | — | `(17,)` | `float32` | `(-inf, inf)` | Exact index mapping below (§1.2). |
| `rays` | `distance` | `(64,)` | `float32` | `[0.0, 1.0]` | `hit_dist / 1024.0` (`1.0` = no hit). Index `0` = straight ahead, CCW (`16`=left, `32`=behind, `48`=right). |
| `rays` | `hit_type` | `(64,)` | `int64` | `{0..5}` | `0: none`, `1: wall`, `2: door`, `3: locked_door`, `4: enemy`, `5: item`. |
| `visible_entities` | `features` | `(32, 5)` | `float32` | `(-inf, inf)` | `[dist/1024, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]`, nearest first (`90°` FOV & LOS). |
| `visible_entities` | `entity_type` | `(32,)` | `int64` | `{0..11}` | `0: pad`, `1: enemy_melee`, `2: enemy_hitscan`, `3: enemy_projectile`, `4: projectile`, `5: health`, `6: armor`, `7: ammo`, `8: weapon`, `9: key`, `10: barrel`, `11: other`. |
| `visible_entities` | `mask` | `(32,)` | `int8`/`bool` | `{0, 1}` | `1` = valid entity slot, `0` = zero-padded empty slot. |

### 1.2 `vitals` Index Table (`VITALS_FIELDS`)
| Index | Field | Scale Normalization in `preprocess()` |
| :--- | :--- | :--- |
| `0` | `health` | `/ 100.0` (clipped `[0, 2.0]`) |
| `1` | `armor` | `/ 100.0` (clipped `[0, 2.0]`) |
| `2` | `selected_weapon` | One-hot `(8,)` over slots `0..7` (or `/ 7.0` + embedding) |
| `3` | `selected_weapon_ammo` | `log1p(clip(x, 0, 500)) / 6.2` |
| `4..7` | `ammo_bullets`, `ammo_shells`, `ammo_rockets`, `ammo_cells` | `log1p(clip(x, 0, 500)) / 6.2` |
| `8..13` | `key_blue_card`, `key_yellow_card`, `key_red_card`, `key_blue_skull`, `key_yellow_skull`, `key_red_skull` | Already `{0.0, 1.0}` |
| `14..16` | `vel_forward`, `vel_right`, `vel_up` | `/ 30.0` (clipped `[-2.0, 2.0]`) |

### 1.3 Official Action Array (`action: np.ndarray`)
* Shape `(9,)`, `dtype=np.int8`, values strictly in `{0, 1}`:
  * `0: MOVE_FORWARD`, `1: MOVE_BACKWARD`, `2: MOVE_LEFT`, `3: MOVE_RIGHT`, `4: TURN_LEFT`, `5: TURN_RIGHT`, `6: ATTACK`, `7: USE`, `8: SELECT_NEXT_WEAPON`.

---

## 2. Preprocessed Neural Inputs (`ModelInputs`) & Memory State

*(For single-step inference `B = 1`; for PPO BPTT training flatten `(B * T, ...)` in encoders then reshape `(B, T, D)` for GRU.)*

### 2.1 `ModelInputs` Tensor Dict / Dataclass
| Field | Shape | Dtype | Construction |
| :--- | :--- | :--- | :--- |
| `rays` | `(B, 8, 64)` | `float32` | Channel 0: `distance`; Channel 1: `1.0 - distance` (proximity); Channels 2–7: 6-way one-hot of `hit_type`. |
| `ent_feats` | `(B, 32, 5)` | `float32` | `[dist, sin_a, cos_a, vel_fwd / 30.0, vel_right / 30.0]`. |
| `ent_types` | `(B, 32)` | `int64` | Raw `entity_type` indices `0..11` (`0 = pad`). |
| `ent_mask` | `(B, 32)` | `bool` | `mask > 0`. |
| `vitals` | `(B, 27)` | `float32` | Normalized 16 scalar vitals + 8-dim weapon one-hot + `prev_action (9)`? Kept modular: `16` scalars + `8` weapon one-hot + `use_cooldown (1)` + `has_any_key (1)` + `ammo_low (1)` = `27` floats, concatenated with `prev_action (B, 9)` → `(B, 36)`. |
| `rgb_depth` | `(B, 4, 48, 64)` | `float32` | `cv2.resize` bilinear `160×120 → 64×48`; channels `0..2 = rgb / 255.0`, channel `3 = depth / 255.0`. Zeroed with prob `p = 0.25` during training (modality dropout). |
| `grid` | `(B, 6, 32, 32)` | `float32` | Egocentric `32×32` local map (`48` map units/cell, agent at center `(16, 16)` facing `+row/up`): `0: visited_decay`, `1: wall`, `2: door`, `3: locked_door`, `4: item_or_key`, `5: enemy_recent`. |
| `action_mask` | `(B, 9)` | `bool` | `True` = button allowed to be `1`; `False` = forced `0` (`logits = -1e8`). `mask[7] (USE) = False` while `use_cooldown_steps > 0`; `mask[6] (ATTACK) = False` when `selected_weapon > 1 and selected_weapon_ammo <= 0`. |

### 2.2 `OdomGridState` (Per-Environment Stateful Preprocessor Tracker)
Reset on `DoomAgent.reset()` and whenever `done == True` in vectorized rollouts:
* `grid`: `np.ndarray` `(6, 32, 32)` `float32`
* `prev_ray_dist`: `np.ndarray` `(64,)` `float32` (or `None` on step 0)
* `prev_ray_type`: `np.ndarray` `(64,)` `int64`
* `prev_action`: `np.ndarray` `(9,)` `int8`
* `use_cooldown`: `int` (counts down `6 → 0` after `action[7] == 1` when any front ray `[-4..+4]` sees `door`/`locked_door` within `dist < 0.15`)

### 2.3 `RecurrentState` (PyTorch Tensor Tuple)
* `gru_h`: `(1, B, 256)` `float32` — GRU hidden state.
* `obj_slots`: `(B, 4, 8)` `float32` — Learned objective memory slots (`0: key_state`, `1: locked_door_mem`, `2: exit_mem`, `3: combat_breadcrumb`), warped by `(Δx, Δy, Δθ)` + gated GRU write.

---

## 3. Human Demo Dataset Schema (`data/demos/<player>/<ep_id>.npz`)

Compressed NumPy archive (`np.savez_compressed`), one file per recorded episode (gitignored from repo):
* `rgb`: `(T, 48, 64, 3)` `uint8` *(already resized to save disk & speed up BC loading!)*
* `depth`: `(T, 48, 64)` `uint8`
* `vitals`: `(T, 17)` `float32`
* `rays_distance`: `(T, 64)` `float32`
* `rays_hit_type`: `(T, 64)` `uint8`
* `ent_features`: `(T, 32, 5)` `float32`
* `ent_type`: `(T, 32)` `uint8`
* `ent_mask`: `(T, 32)` `uint8`
* `actions`: `(T, 9)` `uint8`
* `rewards`: `(T,)` `float32`
* `dones`: `(T,)` `bool` (`True` on final step `T-1`)
* `meta_json`: scalar UTF-8 JSON string:
  ```json
  {"wad": "tier3_s104.wad", "map": "MAP01", "tier": 3, "player": "swapnil", "exit_reached": true, "ticks": 1480, "contract_version": "1.0.0-draft"}
  ```

---

## 4. Checkpoint Schema (`.pt`)

Saved via `torch.save(payload, path)` and loaded with `weights_only=False` (or safetensors + JSON config):
```python
{
    "contract_version": "1.0.0-draft",
    "git_sha": "a1b2c3d",
    "global_step": 18_500_000,
    "update_idx": 903,
    "model_config": {
        "d_model": 256,
        "gru_dim": 256,
        "num_slots": 4,
        "slot_dim": 8,
        "use_rgb": True,
        "use_grid": True,
    },
    "model_state_dict": model.state_dict(),
    "scorer_summary": {          # Populated by LOQ judge upon promotion
        "total_score": 74.8,
        "mean_act_ms": 14.2,
        "p99_act_ms": 23.9,
    },
}
```

---

## 5. Map Manifest, PLR & Scorer JSON Schemas

### 5.1 Map Manifest (`training/scorer/eval_suite.json`)
```json
[
  {
    "id": "heldout_t1_arena_01",
    "wad": "maps/heldout/t1_arena_01.wad",
    "map_name": "MAP01",
    "tier": 1,
    "episode_ticks": 2100,
    "seed": 1001,
    "source": "procedural_heldout",
    "split": "heldout"
  }
]
```
*(Rule: Training procedural seeds use `seed ∈ [100_000, 999_999]`; held-out procedural seeds use `seed ∈ [1_000, 9_999]`.)*

### 5.2 PLR Buffer Entry (In-Memory / Checkpoint Dict)
```json
{
  "level_id": "proc_t3_s240912",
  "tier": 3,
  "knobs": {"num_rooms": 5, "num_keys": 1, "enemies_per_room": 3, "ammo_mult": 1.0},
  "seed": 240912,
  "score_gae": 0.412,
  "last_seen_update": 310,
  "visits": 4
}
```

### 5.3 Scorer Output (`runs/eval/<ckpt_name>_score.json`)
```json
{
  "checkpoint": "ckpt_step_018500000.pt",
  "git_sha": "a1b2c3d",
  "total_score": 74.82,
  "tier_scores": {"1": 88.5, "2": 79.1, "3": 68.4, "4": 63.2},
  "exit_rate": 0.625,
  "progression_rate": 0.71,
  "avg_sector_coverage": 0.78,
  "avg_kills": 6.4,
  "avg_damage_taken": 38.0,
  "mean_act_ms": 14.2,
  "p99_act_ms": 23.9,
  "peak_ram_mb": 1120.0,
  "peak_vram_mb": 420.0,
  "passed_limits": true
}
```

---

## 6. W&B Metric Naming Convention

* **Throughput & Perf:** `perf/sps`, `perf/update_time_s`, `perf/act_ms_mean`, `perf/act_ms_p99`
* **Rollout Stats (Train Envs):** `rollout/ep_return`, `rollout/exit_rate`, `rollout/sector_coverage`, `rollout/kills`, `rollout/keys_picked`, `rollout/doors_unlocked`
* **Losses:** `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/bc_kl_loss`, `train/rnd_loss`, `train/aux_loss`, `train/approx_kl`, `train/grad_norm`
* **Curriculum:** `plr/tier_mix_1..4`, `plr/scaffolding_alpha`, `plr/bc_kl_beta`
* **Held-Out Judge (LOQ):** `eval/total_score`, `eval/exit_rate`, `eval/sector_coverage`, `eval/tier1_score`, `eval/tier2_score`, `eval/tier3_score`, `eval/tier4_score`

---

## 7. Default Hyperparameter Configs (`training/configs/`)

```yaml
# training/configs/ppo_curriculum.yaml
seed: 42
num_envs: 16                 # Tune to (OMEN physical/logical cores - 2)
rollout_steps: 128           # 16 * 128 = 2,048 transitions per PPO update
bptt_len: 32                 # 4 chunks per env per rollout; 64 chunks total
ppo_epochs: 4
minibatch_chunks: 16         # 16 * 32 = 512 steps per minibatch
gamma: 0.99
gae_lambda: 0.95
clip_coef: 0.2
lr: 2.5e-4
max_grad_norm: 0.5
vf_coef: 0.5
ent_coef: 0.01               # Anneal 0.015 -> 0.005
rnd_coef: 0.05
grid_novelty_coef: 0.03
aux_coef: 0.10
bc_kl_beta_init: 0.20        # Linear decay -> 0.0 over 4,000,000 env steps
rgb_dropout_p: 0.25          # Zero RGB+depth tensor per sequence chunk with p=0.25
scaffolding_end_step: 12000000
tier_weights: [1.0, 1.5, 2.5, 3.0]
```
