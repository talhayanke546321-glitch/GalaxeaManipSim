"""Audit a standard-R1 task without collecting data or loading a policy.

Run one task per process so SAPIEN/CoACD asset loading is isolated.  The
script reports the observation/action contract, object metadata, task prompt,
Gym time limit, and reset reproducibility as JSON.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import gymnasium as gym
import numpy as np

import galaxea_sim.envs  # noqa: F401 - registers the Gym environments


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


def _state_action(observation: dict) -> np.ndarray:
    upper = observation["upper_body_observations"]
    return np.concatenate(
        [
            upper["left_arm_joint_position"],
            upper["left_arm_gripper_position"],
            upper["right_arm_joint_position"],
            upper["right_arm_gripper_position"],
        ]
    ).astype(np.float32)


def _object_difference(first: dict, second: dict) -> tuple[bool, float | None]:
    if first.keys() != second.keys():
        return False, None

    max_difference = 0.0
    for key in first:
        first_value = np.asarray(first[key])
        second_value = np.asarray(second[key])
        if first_value.shape != second_value.shape:
            return False, None
        if not np.issubdtype(first_value.dtype, np.number):
            if not np.array_equal(first_value, second_value):
                return False, None
            continue
        max_difference = max(
            max_difference,
            float(np.max(np.abs(first_value - second_value), initial=0.0)),
        )
    return bool(max_difference <= 1e-7), max_difference


def audit_task(env_name: str, seed: int, perturb_steps: int) -> dict:
    env = gym.make(
        env_name,
        headless=True,
        obs_mode="state",
        controller_type="bimanual_joint_position",
    )
    try:
        first_observation, first_reset_info = env.reset(seed=seed)
        raw_env = env.unwrapped
        first_objects = first_observation["object_dict"]
        first_action = _state_action(first_observation)

        action = first_action.copy()
        action[0] += 0.04
        last_step = None
        for _ in range(perturb_steps):
            last_step = env.step(action)

        qvel_before_reset = np.asarray(raw_env.robot.get_qvel()).copy()
        second_observation, second_reset_info = env.reset(seed=seed)
        qvel_after_reset = np.asarray(raw_env.robot.get_qvel()).copy()
        objects_match, max_object_difference = _object_difference(
            first_objects, second_observation["object_dict"]
        )

        second_action = _state_action(second_observation)
        result = {
            "env_name": env_name,
            "registered_max_episode_steps": getattr(env.spec, "max_episode_steps", None),
            "action_space_shape": list(env.action_space.shape),
            "packed_state_shape": list(second_action.shape),
            "language_instruction": second_observation.get("language_instruction", ""),
            "object_keys": sorted(second_observation["object_dict"].keys()),
            "first_reset_info": first_reset_info,
            "second_reset_info": second_reset_info,
            "same_seed_objects_match": objects_match,
            "same_seed_object_max_abs_difference": max_object_difference,
            "same_seed_robot_state_matches": bool(
                np.allclose(first_action, second_action, rtol=0.0, atol=1e-7)
            ),
            "qvel_norm_before_reset": float(np.linalg.norm(qvel_before_reset)),
            "qvel_norm_after_reset": float(np.linalg.norm(qvel_after_reset)),
            "qvel_cleared_by_reset": bool(
                np.allclose(qvel_after_reset, 0.0, rtol=0.0, atol=1e-7)
            ),
        }
        if last_step is not None:
            _, reward, terminated, truncated, info = last_step
            result["probe_step"] = {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        return _json_value(result)
    finally:
        env.close()


def audit_openpi_contract(env_name: str, seed: int) -> dict:
    # Import lazily so the base simulator audit stays usable without the
    # optional OpenPI client package.
    from galaxea_sim.integrations.openpi.environment import GalaxeaSimEnvironment

    env = GalaxeaSimEnvironment(env_name=env_name, seed=seed, headless=True)
    try:
        env.reset()
        observation = env.get_observation()
        return {
            "env_name": env_name,
            "max_episode_steps": env.max_episode_steps,
            "state_shape": list(observation["state"].shape),
            "prompt": observation["prompt"],
            "image_shapes": {
                key: list(value.shape) for key, value in observation["images"].items()
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturb-steps", type=int, default=3)
    parser.add_argument("--openpi-contract", action="store_true")
    args = parser.parse_args()
    if args.openpi_contract:
        result = audit_openpi_contract(args.env_name, args.seed)
    else:
        result = audit_task(args.env_name, args.seed, args.perturb_steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
