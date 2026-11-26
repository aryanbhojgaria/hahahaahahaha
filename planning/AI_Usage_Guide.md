# Antigravity Model & Slash-Command Playbook (1 Paid Pro + 3 Free Accounts)

## 1. Quota Strategy Across the Team

* **Vedesh (OMEN) has Paid Google AI Pro:** Highest quota for **Gemini Pro (High thinking)** and **Claude Opus** in Antigravity.
* **Aryan (LOQ), Aman (M5), Swapnil (M3) are on Free Tiers:** Limited daily/hourly turns on heavy reasoning models (**Gemini Pro High / Claude Opus**), generous turns on **Gemini Flash**.

### Golden Rule of Model Selection
1. **Default to `Gemini Flash` (~75–80% of turns on Free accounts):** Use Flash for boilerplate, data loaders, config YAML files, `pytest` unit tests, WAD lump packing tables, `pygame` keyboard capture, W&B logging wrappers, and Markdown/README drafting.
2. **Save `Gemini Pro (High thinking)` / `Claude Opus` for Hard Algorithmic Spikes (~20–25% of turns):**
   * **Vedesh (Paid):** Keep **Gemini Pro (High)** or **Claude Opus** on for `ppo.py` (GAE + truncated BPTT hidden-state masking + BC-KL), debugging RL training instabilities, and reviewing PRs before merging to `main`.
   * **Aryan (Free — ration carefully!):** Use **Gemini Pro (High)** / **Claude Opus** *only* for the two mathematically subtle files: `odometry_grid.py` (circular scan-matching + affine warp) and `model.py` (masked entity attention + slot updates). Switch immediately to **Gemini Flash** for `score_checkpoint.py`, latency scripts, and tests.
   * **Aman (Free — ration carefully!):** Use **Gemini Pro (High)** *only* for designing the Tier 3 keycard-lock graph topology (`tiers.py`) and `curriculum_plr.py`. Use **Gemini Flash** for WAD byte/UDMF builders, texture lists, and `rewards.py`.
   * **Swapnil (Free):** Use **Gemini Flash** for `record_demos.py`, unit tests, replay video stitching, and initial README drafts; spend your limited **Gemini Pro (High)** turns polishing the **2-Page Project Brief PDF** prose and auditing rule compliance.
3. **Cross-Team Escalation Trick:** If Aryan, Aman, or Swapnil hits a Free-tier Pro rate limit on a gnarly math/RL bug, push the branch and let **Vedesh pull the branch on OMEN and run `/boost` using his Paid AI Pro quota!**

---

## 2. Task-to-Model Quick Reference Table

| Module / Task | Owner | Recommended Model | Recommended Slash Command |
| :--- | :--- | :--- | :--- |
| Freeze interfaces & edge cases at H0 | Vedesh + Aryan | Gemini Pro (High) | `/grill-me` |
| `training/envs.py` subprocess workers | Vedesh | Gemini Flash → Pro review | `/plan` |
| `training/ppo.py` (GAE + BPTT + BC-KL) & `rnd.py` | Vedesh | **Gemini Pro (High) / Claude Opus** | `/boost` |
| `training/odometry_grid.py` (360° scan-match + warp) | Aryan | **Gemini Pro (High) / Claude Opus** | `/boost` |
| `training/model.py` & `submission/agent.py` | Aryan | **Gemini Pro (High)** | `/plan` |
| `training/scorer/score_checkpoint.py` & tests | Aryan | **Gemini Flash** | `/plan` |
| Continuous LOQ checkpoint scoring every 30m | Aryan | Gemini Flash | `/schedule` |
| `training/mapgen/` (UDMF/WAD + Tier 1–4 graphs) | Aman | Pro for graph logic, **Flash** for WAD I/O | `/plan` then `/goal` |
| Stress-generate & validate 1,000 solvable WADs | Aman | **Gemini Flash** | `/goal` |
| `training/rewards.py` & `curriculum_plr.py` | Aman | Gemini Pro (High) / Flash | `/plan` |
| Finding permissive community practice WADs | Aman | Gemini Flash | `/browser` |
| `training/bc/record_demos.py` (`pygame` UI) | Swapnil | **Gemini Flash** | `/plan` |
| `tests/*.py` unit test suites | Swapnil | **Gemini Flash** | `/plan` |
| 2-Page Brief PDF, Architecture SVG & Q&A prep | Swapnil | **Gemini Pro (High)** (final polish) | `/plan` |
| Diagnosing stalled PPO entropy / reward collapse | Vedesh | **Gemini Pro (High) / Claude Opus** | `/boost` |

---

## 3. Antigravity Slash-Command Playbook

* **`/plan` — Use Before Every New Module:** Type `/plan` before writing `ppo.py`, `model.py`, `tiers.py`, or `score_checkpoint.py`. Reviewing a 10-line plan catches tensor-shape mismatches (`Schema.md`) before wasting tokens rewriting 300 lines of code.
* **`/boost` — Reserve for the 3 Hardest Problems:** Triggers deep multi-angle reasoning and rigorous verification. Use it **only** on:
  1. `training/ppo.py` (recurrent BPTT mask & GAE alignment — silent bugs here ruin RL),
  2. `training/odometry_grid.py` (coordinate-frame signs on scan-matching + `warpAffine`), and
  3. Diagnosing mid-event RL plateaus (paste W&B metrics on Vedesh's paid Pro account).
* **`/goal` — Unattended Thorough Execution:** Ideal when you have a clear automated pass criterion:
  * *Aman:* `"/goal Generate 200 maps per tier (800 total), run BFS solvability + doom_env reset/step smoke test on every map, and fix any sector/node bugs until 100% pass."`
  * *Aryan:* `"/goal Profile DoomAgent.act under torch.set_num_threads(1) and optimize until mean latency < 18ms and validate.py --docker passes."`
* **`/schedule` — Live Hackathon Automation:**
  * *On Aryan's LOQ (H6–H20):* `"/schedule Every 30 minutes check runs/incoming/ for new checkpoints from OMEN, run score_checkpoint.py + validate.py, log to W&B, and alert if a new best total_score is achieved."`
  * *On Vedesh's OMEN (overnight H8–H16):* `"/schedule Every 60 minutes inspect the latest training log / W&B metrics for NaNs, worker crashes, or collapsed entropy (< 0.3) and summarize run health."`
* **`/browser` — Fast Asset & Doc Lookup:** Use on Aman's or Swapnil's laptop to locate permissive community Doom WAD archives (`idgames` single-map vanilla/Boom WADs `< 2048` sectors) or verify UDMF linedef lock special numbers.
* **`/grill-me` — Quick Alignment on Ambiguities:** Use at H0 if organizers drop updated `CONTRACT_VERSION` notes or final VRAM/CPU numbers.
* **`/learn` — Persist Tricky Machine Quirks:** Once WSL2 + `vizdoom==1.3.3` + `zdbsp` paths or `validate.py --docker` flags are working on OMEN/LOQ, run `/learn` so Antigravity remembers exact binary paths and `uv run` invocations across every new chat session!
* **`/compact` — Save Context & Quota:** Run `/compact` whenever a coding thread exceeds ~25–30 turns so responses stay fast and Free-tier token limits last all 24 hours.
* **Why We SKIP `/teamwork`:** Four humans are already working in parallel across four physical machines. Spawning autonomous multi-agent swarms inside one repo creates merge conflicts and burns Free-tier quotas rapidly. Stick to **1 human + 1 focused Antigravity session per branch**.
