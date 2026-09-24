"""Run one built-in expert solution and report whether the task completes."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import gymnasium as gym
import numpy as np

import galaxea_sim.envs  # noqa: F401 - registers environments
from galaxea_sim.planners.bimanual import BimanualPlanner


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def run_expert(env_name: str, seed: int, max_steps: int | None) -> dict:
    started = time.monotonic()
    make_kwargs = {"headless": True, "obs_mode": "state", "control_freq": 15}
    if max_steps is not None:
        # Gym handles this keyword itself and overrides the registered
        # TimeLimit without forwarding it to the simulator constructor.
        make_kwargs["max_episode_steps"] = max_steps
    env = gym.make(env_name, **make_kwargs)
    try:
        raw_env = env.unwrapped
        planner = BimanualPlanner(
            urdf_path=f"{raw_env.robot.name}/robot.urdf",
            srdf_path=None,
            left_arm_move_group=raw_env.left_ee_link_name,
            right_arm_move_group=raw_env.right_ee_link_name,
            active_joint_names=raw_env.active_joint_names,
            control_freq=raw_env.control_freq,
        )
        _, reset_info = env.reset(seed=seed)
        registered_limit = getattr(env.spec, "max_episode_steps", None)
        step_limit = max_steps or registered_limit or 1000
        steps = 0
        substeps = 0
        planning_failures = 0
        failed_substeps: list[dict] = []
        terminated = False
        truncated = False
        info: dict = {}

        for substep in raw_env.solution():
            substeps += 1
            actions = planner.solve(
                substep,
                raw_env.robot.get_qpos(),
                raw_env.last_gripper_cmd,
                verbose=False,
            )
            if actions is None:
                planning_failures += 1
                failed_substeps.append(
                    {"index": substeps, "operation": str(substep[0]) if substep else "unknown"}
                )
                continue
            for action in actions:
                _, _, terminated, truncated, info = env.step(action)
                steps += 1
                if bool(info.get("success")) or terminated or truncated or steps >= step_limit:
                    break
            if bool(info.get("success")) or terminated or truncated or steps >= step_limit:
                break

        return _json_value(
            {
                "env_name": env_name,
                "seed": seed,
                "registered_max_episode_steps": registered_limit,
                "steps": steps,
                "solution_substeps": substeps,
                "planning_failures": planning_failures,
                "failed_substeps": failed_substeps,
                "success": bool(info.get("success", False)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "hit_step_limit": steps >= step_limit,
                "reset_info": reset_info,
                "final_objects": raw_env.get_object_dict(),
                "final_info": info,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    print(json.dumps(run_expert(args.env_name, args.seed, args.max_steps), indent=2))


if __name__ == "__main__":
    main()
