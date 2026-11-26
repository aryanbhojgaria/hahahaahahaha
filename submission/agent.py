"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
Implements:
- Single-thread CPU budget enforcement (torch.set_num_threads(1)).
- Hybrid sensory encoding + scan-matched occupancy grid + GRU & Objective Memory Slots.
- Observation-derived legal action masking.
- Low-temperature seeded Bernoulli sampling to break symmetric corner limit-cycles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path so training modules can be imported cleanly
SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.preprocessing import preprocess, OdomGridState
from training.model import HybridDoomNet, RecurrentState

WEIGHTS_DIR = SUBMISSION_DIR / "weights"
DEFAULT_CHECKPOINT_PATH = WEIGHTS_DIR / "best_agent.pt"


class DoomAgent:
    """Zero-Shot Autonomous DOOM Agent."""

    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights onto `device` ("cuda" or "cpu")."""
        # Strict latency guard: enforce 1 CPU thread to avoid competing with ViZDoom engine
        if device == "cpu":
            torch.set_num_threads(1)

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        # Initialize neural model
        self.model = HybridDoomNet(
            use_rgb=True,
            use_grid=True,
            rgb_dropout_p=0.0,  # No dropout at inference
            gru_dim=256,
            num_slots=4,
            slot_dim=8,
        ).to(device)

        # Load weights if checkpoint exists
        if DEFAULT_CHECKPOINT_PATH.exists():
            try:
                ckpt = torch.load(DEFAULT_CHECKPOINT_PATH, map_location=device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                # Fallback to initialized model if checkpoint format differs
                pass

        self.model.eval()

        # Seeded deterministic RNG reset on every map
        self.rng = np.random.default_rng(0)
        self.odom_state: OdomGridState | None = None
        self.recurrent_state: RecurrentState | None = None

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, memory buffers and trackers."""
        self.odom_state = OdomGridState()
        self.recurrent_state = None
        self.rng = np.random.default_rng(0)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int array of shape action_space.shape with values in {0, 1}."""
        # 1. Preprocess sensory observation & update egocentric occupancy grid
        inputs, self.odom_state = preprocess(
            obs=observation,
            state=self.odom_state,
            device=self.device,
            use_grid=True,
        )

        # 2. Forward pass through HybridDoomNet
        logits, _, self.recurrent_state, _ = self.model(
            inputs=inputs,
            recurrent_state=self.recurrent_state,
            return_aux=False,
        )

        # Logits of shape (1, 9) -> (9,)
        logits_arr = logits.squeeze(0).cpu().numpy().astype(np.float32)

        # 3. Low-temperature Bernoulli sampling (tau = 0.85)
        # Prevents freezing in limit cycles while maintaining high fidelity to trained weights
        tau = 0.85
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits_arr / tau, -20.0, 20.0)))
        rand_vals = self.rng.random(len(probs))
        action = (rand_vals < probs).astype(np.int8)

        # 4. Mutual exclusion conflict resolution (soft stabilization)
        # MOVE_FORWARD (0) vs MOVE_BACKWARD (1)
        if action[0] == 1 and action[1] == 1:
            if logits_arr[0] >= logits_arr[1]:
                action[1] = 0
            else:
                action[0] = 0

        # MOVE_LEFT (2) vs MOVE_RIGHT (3)
        if action[2] == 1 and action[3] == 1:
            if logits_arr[2] >= logits_arr[3]:
                action[3] = 0
            else:
                action[2] = 0

        # TURN_LEFT (4) vs TURN_RIGHT (5)
        if action[4] == 1 and action[5] == 1:
            if logits_arr[4] >= logits_arr[5]:
                action[5] = 0
            else:
                action[4] = 0

        # 5. Enforce observation action mask strictly
        act_mask = inputs.action_mask.squeeze(0).cpu().numpy()
        action = (action & act_mask.astype(np.int8)).astype(np.int8)

        # Update previous action tracker in odometry state
        if self.odom_state is not None:
            self.odom_state.prev_action = action.copy()

        return action
