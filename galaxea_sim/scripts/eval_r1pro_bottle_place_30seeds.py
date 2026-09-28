"""Evaluate the trained R1 Pro bottle-place policy over fixed random seeds.

The evaluator keeps the policy server and the simulator in separate processes,
uses the same 15/10 action-chunk contract as deployment, and assigns seeds to
the six table heights used by the demonstration set in round-robin order.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import tyro
from loguru import logger as loguru_logger
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy

import galaxea_sim.envs  # noqa: F401 - register Gym environments
from galaxea_sim import ASSETS_DIR
from galaxea_sim.integrations.openpi.r1_pro_action_adapter import GalaxeaR1ProActionAdapter
from galaxea_sim.integrations.openpi.r1_pro_chunk_contract import (
    R1ProActionChunkContractPolicy,
    validate_r1_pro_policy_metadata,
)
from galaxea_sim.integrations.openpi.r1_pro_contract import (
    ACTION_DIM,
    EXECUTE_ACTION_HORIZON,
    POLICY_ACTION_HORIZON,
)
from galaxea_sim.integrations.openpi.r1_pro_observation_adapter import (
    GalaxeaR1ProObservationAdapter,
)
from galaxea_sim.integrations.openpi.r1_pro_safety import make_r1_pro_safety_filter
from galaxea_sim.utils.render_environment import prepare_render_environment


STATIC_COLLISION_NAMES = frozenset({"ground", "table", "wall"})
DEFAULT_HEIGHTS = "0.900,0.950,1.000,1.025,1.0375,1.050"


@dataclasses.dataclass
class Args:
    env_name: str = "R1ProBottlePickPlace-v0"
    seed_start: int = 0
    num_episodes: int = 30
    # Round-robin assignment means 30 episodes give five seeds per height.
    table_heights: str = DEFAULT_HEIGHTS
    gui: bool = False
    control_freq: int = 15
    max_episode_steps: int | None = None
    execute_horizon: int = EXECUTE_ACTION_HORIZON
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    api_key: str | None = None
    max_joint_delta: float = 0.12
    max_gripper_delta: float = 0.01
    output: str = "evaluations/pi05_r1pro_bottle_place_30seeds.json"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _entity_name(entity: Any) -> str:
    if entity is None:
        return ""
    if isinstance(entity, str):
        return entity
    name = getattr(entity, "name", None)
    if name:
        return str(name)
    get_name = getattr(entity, "get_name", None)
    if get_name is not None:
        try:
            name = str(get_name())
            if name:
                return name
        except Exception:  # pragma: no cover - SAPIEN binding variation
            pass
    owner = getattr(entity, "entity", None)
    if owner is None:
        get_entity = getattr(entity, "get_entity", None)
        if get_entity is not None:
            try:
                owner = get_entity()
            except Exception:  # pragma: no cover - SAPIEN binding variation
                owner = None
    return _entity_name(owner) if owner is not None and owner is not entity else ""


def _unexpected_robot_contacts(scene: Any, robot_link_names: set[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for contact in scene.get_contacts():
        entities = (getattr(contact, "entity0", None), getattr(contact, "entity1", None))
        names = tuple(_entity_name(entity) for entity in entities)
        if not all(names):
            bodies = getattr(contact, "bodies", ())
            if len(bodies) == 2:
                names = tuple(_entity_name(body) for body in bodies)
        if len(names) != 2 or not all(names):
            continue
        robot_name = names[0] if names[0] in robot_link_names else names[1] if names[1] in robot_link_names else None
        other_name = names[1] if robot_name == names[0] else names[0] if robot_name == names[1] else None
        if robot_name is None or other_name not in STATIC_COLLISION_NAMES:
            continue
        if other_name == "ground" and robot_name.startswith("wheel_motor_link"):
            continue
        pairs.add(tuple(sorted((robot_name, other_name))))
    return pairs


def _parse_heights(text: str) -> list[float]:
    heights = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not heights:
        raise ValueError("table_heights must contain at least one height")
    if any(not 0.70 <= height <= 1.10 for height in heights):
        raise ValueError(f"table heights must be in [0.70, 1.10], got {heights}")
    return heights


def _task_info(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "success",
        "placed",
        "released",
        "placement_hold_steps",
        "placement_xy_error",
        "placement_z_error",
        "placement_tilt_deg",
        "table_height_m",
    )
    return {key: _json_value(info[key]) for key in keys if key in info}


def _failure_reason(*, success: bool, task_info: dict[str, Any], collision: bool, clipped: int, rejected: int, timeout: bool) -> str:
    if success:
        return "success"
    if rejected:
        return "safety_rejection"
    if collision:
        return "collision"
    if clipped:
        return "joint_limit_clipping"
    if timeout:
        return "timeout"
    if task_info.get("released") is False:
        return "not_released"
    if task_info.get("placed") is False:
        return "placement_condition_not_met"
    return "task_success_condition_not_met"


def _run_episode(
    env: gym.Env,
    raw_env: Any,
    broker: action_chunk_broker.ActionChunkBroker,
    observation_adapter: GalaxeaR1ProObservationAdapter,
    action_adapter: GalaxeaR1ProActionAdapter,
    safety_filter: Any,
    *,
    seed: int,
    table_height: float,
    control_freq: int,
    max_episode_steps: int,
    gui: bool = False,
) -> dict[str, Any]:
    observation, reset_info = env.reset(seed=seed)
    # ``env.step`` advances SAPIEN and updates the render scene, but the
    # interactive viewer is only created/refreshed by ``env.render()``.  Call
    # it explicitly in GUI mode so --gui actually opens and updates a window.
    if gui:
        env.render()
    broker.reset()
    robot_link_names = {str(link.name) for link in raw_env.robot.links}
    collision_pairs_seen: set[tuple[str, str]] = set()
    previous_collision_pairs: set[tuple[str, str]] = set()
    collision_contact_steps = 0
    collision_event_count = 0
    clipped = 0
    rate_limited = 0
    rejected = 0
    steps = 0
    done = False
    last_info: dict[str, Any] = {}

    while not done and steps < max_episode_steps:
        policy_result = broker.infer(observation_adapter(observation))
        candidate = action_adapter(policy_result)
        current_state = observation_adapter.state_from_observation(observation)
        safe_action, report = safety_filter.filter(candidate, current_state)
        clipped += int(report.joint_limit_clipped)
        rate_limited += int(report.rate_limited)
        rejected += int(report.rejected)
        observation, _, terminated, truncated, last_info = env.step(safe_action)
        if gui:
            env.render()
        steps += 1
        done = bool(terminated or truncated)
        pairs = _unexpected_robot_contacts(raw_env._scene, robot_link_names)
        if pairs:
            collision_contact_steps += 1
            collision_pairs_seen.update(pairs)
            collision_event_count += len(pairs - previous_collision_pairs)
        previous_collision_pairs = pairs

    success = bool(last_info.get("success", False))
    task_info = _task_info(last_info)
    timeout = not success and steps >= max_episode_steps
    collision = bool(collision_pairs_seen)
    return {
        "seed": seed,
        "table_height_m": table_height,
        "success": success,
        "episode_steps": steps,
        "duration_seconds": steps / control_freq,
        "timed_out": timeout,
        "failure_reason": _failure_reason(
            success=success,
            task_info=task_info,
            collision=collision,
            clipped=clipped,
            rejected=rejected,
            timeout=timeout,
        ),
        "collision": collision,
        "collision_event_count": collision_event_count,
        "collision_contact_steps": collision_contact_steps,
        "collision_pairs": [list(pair) for pair in sorted(collision_pairs_seen)],
        "out_of_bounds_count": clipped,
        "rate_limited_count": rate_limited,
        "safety_rejection_count": rejected,
        "reset_info": _json_value(reset_info),
        "task_info": task_info,
    }


def _make_env(args: Args, table_height: float) -> tuple[gym.Env, Any, int]:
    env = gym.make(
        args.env_name,
        control_freq=args.control_freq,
        headless=not args.gui,
        obs_mode="image",
        ray_tracing=False,
        include_depth=False,
        camera_resolution_scale=4,
        controller_type="bimanual_joint_position",
        table_height_override=table_height,
    )
    raw_env = env.unwrapped
    if raw_env.action_space.shape != (ACTION_DIM,):
        env.close()
        raise ValueError(f"Expected a {ACTION_DIM}-D R1 Pro action space, got {raw_env.action_space}")
    if raw_env.controller_type != "bimanual_joint_position":
        env.close()
        raise ValueError("The evaluator requires bimanual_joint_position")
    max_steps = args.max_episode_steps or getattr(env.spec, "max_episode_steps", None)
    if max_steps is None or max_steps <= 0:
        env.close()
        raise ValueError("The environment has no positive episode limit")
    return env, raw_env, int(max_steps)


def main(args: Args) -> None:
    prepare_render_environment(require_display=args.gui)
    if args.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if args.execute_horizon != EXECUTE_ACTION_HORIZON:
        raise ValueError(f"R1 Pro evaluation requires execute_horizon={EXECUTE_ACTION_HORIZON}")
    if args.control_freq <= 0:
        raise ValueError("control_freq must be positive")
    heights = _parse_heights(args.table_heights)

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
        api_key=args.api_key,
    )
    metadata = client.get_server_metadata()
    logging.info("R1 Pro policy metadata: %s", metadata)
    validate_r1_pro_policy_metadata(metadata)
    policy = R1ProActionChunkContractPolicy(client)
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=args.execute_horizon)
    observation_adapter = GalaxeaR1ProObservationAdapter(
        default_prompt="pick up the bottle and place it on the plate"
    )
    action_adapter = GalaxeaR1ProActionAdapter()

    seeds = list(range(args.seed_start, args.seed_start + args.num_episodes))
    assignments = [(seed, heights[index % len(heights)]) for index, seed in enumerate(seeds)]
    episodes: list[dict[str, Any]] = []
    safety_filter = None
    for height in heights:
        group = [(seed, assigned) for seed, assigned in assignments if assigned == height]
        if not group:
            continue
        env, raw_env, max_steps = _make_env(args, height)
        try:
            safety_filter = make_r1_pro_safety_filter(
                Path(ASSETS_DIR) / raw_env.robot.urdf_path,
                max_joint_delta=args.max_joint_delta,
                max_gripper_delta=args.max_gripper_delta,
            )
            for index, (seed, assigned_height) in enumerate(group, start=1):
                print(
                    f"Evaluating seed {seed} at table {assigned_height:.4f} m "
                    f"({len(episodes) + 1}/{len(assignments)})",
                    flush=True,
                )
                result = _run_episode(
                    env,
                    raw_env,
                    broker,
                    observation_adapter,
                    action_adapter,
                    safety_filter,
                    seed=seed,
                    table_height=assigned_height,
                    control_freq=args.control_freq,
                    max_episode_steps=max_steps,
                    gui=args.gui,
                )
                episodes.append(result)
                print(
                    f"  success={result['success']} steps={result['episode_steps']} "
                    f"reason={result['failure_reason']} "
                    f"tilt={result['task_info'].get('placement_tilt_deg', 'n/a')}",
                    flush=True,
                )
        finally:
            env.close()

    episodes.sort(key=lambda item: int(item["seed"]))
    success_count = sum(bool(item["success"]) for item in episodes)
    height_summary: dict[str, dict[str, Any]] = {}
    for height in heights:
        group = [item for item in episodes if float(item["table_height_m"]) == height]
        if not group:
            continue
        height_summary[f"{height:.4f}"] = {
            "episodes": len(group),
            "success_count": sum(bool(item["success"]) for item in group),
            "success_rate_percent": 100.0 * sum(bool(item["success"]) for item in group) / len(group),
            "mean_tilt_deg": float(
                np.mean([
                    float(item["task_info"]["placement_tilt_deg"])
                    for item in group
                    if "placement_tilt_deg" in item["task_info"]
                ] or [float("nan")]
            )),
        }
    summary = {
        "env_name": args.env_name,
        "checkpoint_evaluation_protocol": "R1 Pro policy server, 15-step chunk / 10-step execution",
        "seeds": seeds,
        "table_heights_m": heights,
        "episodes": len(episodes),
        "success_count": success_count,
        "success_rate_percent": 100.0 * success_count / len(episodes),
        "collision_episode_count": sum(bool(item["collision"]) for item in episodes),
        "out_of_bounds_count": sum(int(item["out_of_bounds_count"]) for item in episodes),
        "safety_rejection_count": sum(int(item["safety_rejection_count"]) for item in episodes),
        "height_summary": height_summary,
        "episodes_detail": episodes,
    }
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_value(summary), indent=2, ensure_ascii=False) + "\n")
    print("\nEvaluation summary:")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes_detail"}, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="WARNING")
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
