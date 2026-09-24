"""Serve a lightweight stats-backed policy for end-to-end smoke testing.

This is intentionally not a pi05 model.  It uses the built-in 14-D
``ur5e_dual`` quantile statistics from ``pi05_base`` to generate random model
actions, then applies the same Galaxea output transform as the real policy.
It is useful when the full checkpoint cannot fit in the available host memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from openpi.policies.galaxea_policy import (
    GALAXEA_ACTION_HORIZON,
    GALAXEA_EXECUTE_HORIZON,
    GALAXEA_STATE_DIM,
    GalaxeaInputs,
    GalaxeaOutputs,
)
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi_client import base_policy


DEFAULT_STATS = Path(
    "/home/vipuser/robotics/openpi-data/openpi-assets/checkpoints/"
    "pi05_base/assets/ur5e_dual"
)


class StatsSmokePolicy(base_policy.BasePolicy):
    def __init__(self, stats_path: Path, action_horizon: int, seed: int) -> None:
        self._stats = normalize.load(stats_path)
        self._q01 = np.asarray(self._stats["actions"].q01, dtype=np.float32)
        self._q99 = np.asarray(self._stats["actions"].q99, dtype=np.float32)
        if self._q01.shape != (GALAXEA_STATE_DIM,) or self._q99.shape != (GALAXEA_STATE_DIM,):
            raise ValueError(f"expected 14-D action stats, got {self._q01.shape} and {self._q99.shape}")
        self._action_horizon = action_horizon
        self._rng = np.random.default_rng(seed)
        self._inputs = GalaxeaInputs()
        self._outputs = GalaxeaOutputs()
        self._logged = False

    def infer(self, obs: dict) -> dict:
        # Run the input adapter so the smoke server validates the same online
        # observation contract as the real pi05 policy.
        canonical = self._inputs(obs)
        state = np.asarray(canonical["state"], dtype=np.float32)

        # Random model-space actions are deliberately bounded by the selected
        # built-in stats.  The simulator-side safety filter then rate-limits
        # them before they reach the robot.
        actions = self._rng.uniform(
            self._q01,
            self._q99,
            size=(self._action_horizon, GALAXEA_STATE_DIM),
        ).astype(np.float32)
        if not self._logged:
            print(
                "stats_smoke: using ur5e_dual q01/q99; "
                f"state_shape={state.shape}, action_shape={actions.shape}",
                flush=True,
            )
            self._logged = True
        return self._outputs({"actions": actions})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=GALAXEA_ACTION_HORIZON)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.action_horizon != GALAXEA_ACTION_HORIZON:
        raise ValueError(f"--action-horizon must be {GALAXEA_ACTION_HORIZON} for the Galaxea contract")

    policy = StatsSmokePolicy(args.stats, args.action_horizon, args.seed)
    server = WebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=args.port,
        metadata={
            "policy": "stats_smoke",
            "stats": str(args.stats),
            "warning": "not pi05 inference; end-to-end transport smoke test only",
            "robot": "r1",
            "controller": "bimanual_joint_position",
            "action_dim": GALAXEA_STATE_DIM,
            "action_horizon": GALAXEA_ACTION_HORIZON,
            "execute_horizon": GALAXEA_EXECUTE_HORIZON,
        },
    )
    print(f"stats_smoke server listening on ws://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
