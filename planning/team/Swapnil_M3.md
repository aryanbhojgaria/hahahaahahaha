# Swapnil (MacBook Air M3 — Joins at +3–4h) — Step-by-Step Runbook

## 1. Role & Ownership Summary
* **Role:** Human-Demo Imitation Lead, Test & QA Suite Owner, Replay Visualizer & **2-Page Project Brief / Finals Storytelling Lead** (**Free Tier — Save Pro Turns for Brief Polish!**).
* **Files You Own:**
  * `training/bc/record_demos.py` (`pygame` interactive `doom_env` human gameplay recorder saving `.npz` episodes per `Schema.md §3`)
  * `training/scorer/render_replay.py` (Side-by-side MP4 renderer showing Agent First-Person RGB + Live 360° Rays + 32×32 Occupancy Grid for Desk Review!)
  * `tests/test_preprocessing_and_odometry.py` & cross-repo integration tests
  * `README.md` (Root project showcase & clean-machine verification guide)
  * **`Project_Brief_2Page.pdf`** (Mandatory universal deliverable worth direct shortlisting & finals impact) + **Desk-Review Q&A Sheet**
* **Interfaces You Provide & Consume (`Schema.md`):**
  * **Consume from Aman (at H4):** Tier 1–3 starter procedural WAD pack + `two_rooms.wad` + `freedoom2` maps.
  * **Provide to Vedesh (by H5.5):** `data/demos/swapnil/*.npz` (`60–100+` clean human episodes across Tiers 1–3) synced via Tailscale `rsync` to OMEN for `train_bc.py`.
  * **Consume from Aryan (at H16–H20):** Final held-out tournament JSON metrics & ablation table (`Random` vs `BC` vs `PPO No-Grid` vs `Full Hybrid + Scan-Match Grid`).

---

## 2. Phase 0: Pre-Event Setup (Do Before Event or Right Before Joining — Zero Project Code!)

Run on **MacBook Air M3 (macOS Terminal)**:
```bash
# 1. Install Homebrew tools, Git LFS & Tailscale
brew install cmake sdl2 boost git git-lfs ffmpeg tailscale
git lfs install

# 2. Install uv + Python 3.13
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.13

# 3. Smoke-test vizdoom==1.3.3 on M3 (delete /tmp/setup-check before H0!)
git clone https://github.com/Reinforce-SST/devforge-doom-template.git /tmp/setup-check
cd /tmp/setup-check
uv sync
uv run python validate.py
cd ~ && rm -rf /tmp/setup-check

# 4. Verify Tailscale connection to Vedesh's OMEN
tailscale status
ssh <vedesh-user>@<omen-tailscale-ip> "echo ready"
```
* **10-Minute Control Warmup:** Practice DOOM keyboard-only movement (`W/S` move, `A/D` strafe, `←/→` turn, `J`/`Ctrl` attack, **single tap `Space` for `USE` — wait 1 second for door to rise, never mash `Space`!**). Clean human demos make Behavior Cloning 3× stronger!

---

## 3. Phase-by-Phase Execution Checklist (Starts Active Work at Phase 3 / H3.5–H4)

### Phase 3 (H3.5–H6): Onboarding, `record_demos.py` & High-Skill Demo Marathon
* **AI Model & Command:** **`Gemini Flash`** + `/plan` *(fast UI script — zero need to spend Pro quota here)*.
1. **H3.5–H4:00 (15-min Fast Onboard):** Clone team GitHub repo, run `uv sync && uv add --dev pygame pytest`, read `Schema.md §3`, checkout `feat/swapnil-bc-recorder`.
2. **H4:00–H4:30:** Build `training/bc/record_demos.py`:
   * Instantiate `doom_env.make_env(wad, map_name, render_buffers=True)` (`FRAME_SKIP = 4`, ~8.75 decisions/sec real-time clock via `pygame.time.Clock().tick(9)`).
   * Render `obs["rgb"]` upscaled `3×` (`480×360`) in a `pygame` window with a small HUD text overlay showing held keys & health from `obs["vitals"]`.
   * Map keys directly to `MultiBinary(9)` (`BUTTONS` order in `Schema.md §1.3`). For `USE` (`Space`) and `SELECT_NEXT_WEAPON` (`Q`), trigger on `KEYDOWN` for **1 step only** so demos never contain accidental multi-step holds or door bounces!
   * Save only successful / clean episodes (press `R` to discard a clumsy death, auto-save `.npz` on exit or strong combat run).
3. **H4:30–H6:00:** Record **60–90 minutes** of crisp gameplay across:
   * **Tier 1:** Circle-strafing enemies, dodging fireballs, picking up health/ammo.
   * **Tier 2:** Opening closed doors cleanly (walk up → single `USE` tap → pause 3 steps → walk through → explore unvisited rooms → hit exit).
   * **Tier 3:** Spot locked door → explore side rooms → grab keycard → backtrack to locked door → unlock & exit!
4. Sync first batch at H5:15 and full batch at H6:00 to OMEN:
   ```bash
   rsync -avz data/demos/ <vedesh-user>@<omen-tailscale-ip>:~/repo/data/demos/
   ```
* **Done Criteria (H6 Gate):** `80+` clean `.npz` episodes on OMEN; Vedesh's `train_bc.py` trains cleanly and beats random by `≥ 2.5×` on LOQ scorer.

### Phase 4 (H6–H8): Extra Tier 3/4 Demos, Architecture Diagrams & Root `README.md`
* **AI Model & Command:** `Gemini Flash` + `/plan`.
1. Record an additional `30 mins` of challenging Tier 3 backtracking & Tier 4 ambush demos (and `freedoom2` `MAP01–MAP02`) so Run #2 has an even richer dataset.
2. Export high-resolution clean vector diagrams (PNG/SVG at 300 DPI) of the **Hybrid + Scan-Matched Grid Architecture** and **5-Stage Training Loop** from `Design.md`.
3. Draft root `README.md` ( crisp project overview, architecture highlights, quick validation commands, repository structure).
* **Done Criteria (H8 Gate):** High-res diagrams ready for Brief; `README.md` v1 merged.

### Phase 5 (H8–H12): Comprehensive `pytest` Regression Suite & Brief Page 1
* **AI Model & Command:** `Gemini Flash` for `pytest` (`/plan`); **`Gemini Pro (High)`** for Brief Page 1 prose polish.
1. Checkout `feat/swapnil-tests`. Write `tests/test_preprocessing_and_odometry.py`:
   * Test `USE` cooldown mask triggers for 6 steps near doors and resets properly.
   * Test `ATTACK` mask triggers only when `selected_weapon > 1` and `ammo == 0` (fist `weapon == 1` still attacks!).
   * Test scan-matching yaw `Δθ` recovers synthetic circular shifts (`k = -3, 0, +4`) on 64-ray arrays within `< 0.5` bin error!
   * Test `.npz` demo schema roundtrip.
2. Draft **Page 1 of the 2-Page Project Brief** (Google Docs / Typst / LaTeX template strictly formatted to 2 pages, 10pt clean typography, two-column or compact layout):
   * **USP Banner**, **Architecture Diagram**, **360° Scan-Matching Egocentric Occupancy Grid + Objective Memory Slots**, and **Masked Set Attention + Rule-Compliant Action Masking**.
* **Done Criteria (H12 Gate):** Full `uv run pytest` suite passes (`20+` tests green); Page 1 layout locked.

### Phase 6 (H12–H16): Live Brain-View Replay Renderer & Brief Page 2 Draft
* **AI Model & Command:** `Gemini Flash` + `/plan`.
1. Checkout `feat/swapnil-visualizer`. Build `training/scorer/render_replay.py`:
   * Runs `DoomAgent` on any held-out WAD and outputs a `960×360` side-by-side MP4 (`ffmpeg` / `cv2.VideoWriter`):
     * **Left panel:** `160×120` First-Person RGB view,
     * **Center panel:** Top-down polar plot of the 64 360° rays color-coded by hit type (wall/door/lock/enemy/item),
     * **Right panel:** Live `32×32` Egocentric Occupancy Grid + Objective Slot bearings updating in real time!
   * *(Why this wins Stage 2 & Finals: Judges literally watch the agent's neural memory map rooms, remember a red locked door behind it, grab the key, and navigate back!)*
2. Draft **Page 2 of the 2-Page Brief** (5-Stage Curriculum diagram, table skeleton for Held-Out Tiers 1–4 + Ablation, Latency/VRAM footprint, Honest Failure Modes).
* **Done Criteria (H16 Gate):** `render_replay.py` generates a crystal-clear MP4 on current best checkpoint; Page 2 ready for final H20 numbers.

### Phase 7 & 8 (H16–H24): Final Metrics Table, PDF Export, Showcase Video & Q&A Drill
* **AI Model & Command:** **`Gemini Pro (High)`** *(spend remaining Pro quota making the 2-Page Brief read like an award-winning systems paper!)*.
1. **H19–H21:** Plug Aryan's final 40-map held-out tournament scores (`Random`, `BC Warm-Start`, `Recurrent PPO (No Grid)`, `Full Hybrid + Grid (Pre-Vanilla)`, `Final Vanilla Fine-Tuned`) + exact single-thread `act()` ms into Brief Page 2.
2. Export **`Project_Brief.pdf`** — **verify page count is strictly `2` pages** (`pdfinfo` / Preview check), crisp diagrams, zero text overflow.
3. Render 3 showcase MP4 clips (`Tier 1 Combat`, `Tier 2 Multi-Room Doors`, `Tier 3 Keycard Backtracking`) using `render_replay.py` on the locked `best_agent.pt` for live Stage 2 Desk Review.
4. **H22–H23:** Run a 15-minute live Q&A drill with Vedesh, Aryan, and Aman using the 10 questions in `Track1_DOOM_Game_Plan.md §10` so everyone answers crisply in `< 30 seconds`.
5. **H23.5:** Double-check repo public visibility + PDF upload alongside Vedesh.

---

## 4. Tests You Must Write (`tests/test_preprocessing_and_odometry.py`)
* `test_scan_match_synthetic_rotation()`: Roll a realistic 64-ray vector by `+3` indices → assert estimated `Δθ` matches `+3 * (2π/64)` within `15%`.
* `test_fist_attack_never_masked()`: Set `vitals[selected_weapon] = 0` or `1` and `ammo = 0` → assert `action_mask[ATTACK] == True`.
* `test_use_cooldown_prevents_double_press()`: Simulate pressing `USE` with `rays.hit_type[0] == 2` and `rays.distance[0] == 0.08` → assert next `6` steps have `action_mask[USE] == False`, then step `7` restores `True`.

---

## 5. If Things Go Wrong (Swapnil's Fallbacks)
* **`pygame` window focus / key-repeat quirk on macOS:** Poll `pygame.key.get_pressed()` for continuous movement/turn keys (`0..6`) and `KEYDOWN` events for edge-triggered `USE (7)` and `SELECT_NEXT_WEAPON (8)`.
* **Joining slightly past H4:** Vedesh starts PPO smoke run on `two_rooms.wad` + Tier 1–2 without BC first; as soon as your `.npz` files arrive at H5.5–H6, Vedesh warm-starts Run #1 immediately.
* **PDF spills onto Page 3:** Cut verbose prose into compact bullet tables — judges skim diagrams, bolded insights, and ablation tables far faster than paragraphs.
