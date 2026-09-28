"""使用固定 seed 批量评测 OpenPI 策略的仿真闭环。

Evaluate a trained OpenPI policy in GalaxeaManipSim over fixed seeds.

The evaluator intentionally drives Gym directly instead of using the generic
OpenPI Runtime so that each episode uses the exact requested reset seed.  It
also records simulator contacts and the final safety-filter report needed for
closed-loop evaluation metrics.
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

import galaxea_sim.envs  # noqa: F401 - registers the Gym environments
from galaxea_sim import ASSETS_DIR
from galaxea_sim.integrations.openpi.action_adapter import GalaxeaActionAdapter
from galaxea_sim.integrations.openpi.chunk_contract import (
    ActionChunkContractPolicy,
    validate_galaxea_policy_metadata,
)
from galaxea_sim.integrations.openpi.contract import (
    ACTION_DIM,
    EXECUTE_ACTION_HORIZON,
    POLICY_ACTION_HORIZON,
)
from galaxea_sim.integrations.openpi.observation_adapter import GalaxeaObservationAdapter
from galaxea_sim.integrations.openpi.safety import make_r1_safety_filter
from galaxea_sim.utils.render_environment import prepare_render_environment


STATIC_COLLISION_NAMES = frozenset({"ground", "table", "wall"})


@dataclasses.dataclass
class Args:
    """批量评测参数。

    与 ``run_pi05_closed_loop`` 不同，这个脚本直接控制 Gym episode，目的
    是精确指定每个 seed、记录碰撞接触和失败原因。它仍然复用同一个
    WebSocket client、15/10 action chunk 协议和 R1 安全过滤器。
    """

    env_name: str = "R1DiverseBottlesPick-v0"
    seed_start: int = 0
    num_episodes: int = 30
    # Open a SAPIEN viewer through the active desktop session. The policy
    # still observes the same camera images; this only adds visualization.
    gui: bool = False
    control_freq: int = 15
    max_episode_steps: int | None = None
    # The project default is 15/10.  The evaluator also permits an explicit
    # 15-step execution variant so the full model chunk can be tested without
    # changing the trained policy or dataset statistics.
    execute_horizon: int = EXECUTE_ACTION_HORIZON

    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    api_key: str | None = None

    max_joint_delta: float = 0.12
    max_gripper_delta: float = 0.01
    # The OpenPI adapter's rate limiter is an optional defensive layer. Keep
    # joint-limit clipping and finite-value checks even when it is disabled.
    rate_limit: bool = False
    prompt: str | None = None
    output: str = "evaluations/pi05_upright_bottles_seed0_30seeds.json"


def _json_value(value: Any) -> Any:
    """把 NumPy/SAPIEN 返回值递归转换成可写入 JSON 的普通 Python 值。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _entity_name(entity: Any) -> str:
    """兼容不同 SAPIEN 版本，从接触对象解析拥有者名称。"""
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
        except Exception:  # pragma: no cover - binding-dependent fallback
            pass

    # SAPIEN 3.0 contacts expose PhysX body components.  Their component
    # names are often empty; the owning Entity carries the link/actor name.
    owner = getattr(entity, "entity", None)
    if owner is None:
        get_entity = getattr(entity, "get_entity", None)
        if get_entity is not None:
            try:
                owner = get_entity()
            except Exception:  # pragma: no cover - binding-dependent fallback
                owner = None
    if owner is not None and owner is not entity:
        return _entity_name(owner)
    return ""


def _contact_pair(contact: Any) -> tuple[str, str]:
    """从一条 PhysX contact 中取得两侧实体名称。"""
    entity0 = getattr(contact, "entity0", None)
    entity1 = getattr(contact, "entity1", None)
    names = (_entity_name(entity0), _entity_name(entity1))
    if all(names):
        return names

    # SAPIEN versions that do not expose entity0/entity1 still expose bodies.
    bodies = getattr(contact, "bodies", ())
    if len(bodies) == 2:
        return _entity_name(bodies[0]), _entity_name(bodies[1])
    return names


def _unexpected_robot_contacts(scene: Any, robot_link_names: set[str]) -> set[tuple[str, str]]:
    """找出应该计入失败指标的机器人静态物体碰撞。

    Return robot/static contact pairs that count as collisions.

    Contacts with bottles are expected during grasping.  Wheel-ground contact
    is also expected.  Arm/base contact with table, wall, or ground is not.
    """

    pairs: set[tuple[str, str]] = set()
    for contact in scene.get_contacts():
        first, second = _contact_pair(contact)
        if not first or not second:
            continue

        robot_name = first if first in robot_link_names else second if second in robot_link_names else None
        other_name = second if robot_name == first else first if robot_name == second else None
        if robot_name is None or other_name not in STATIC_COLLISION_NAMES:
            continue
        if other_name == "ground" and robot_name.startswith("wheel_motor_link"):
            continue
        pairs.add(tuple(sorted((robot_name, other_name))))
    return pairs


def _final_task_info(info: dict[str, Any]) -> dict[str, Any]:
    """提取任务成功判断和左右臂诊断字段。"""
    keys = (
        "success",
        "left_success",
        "right_success",
        "left_distance",
        "right_distance",
        "left_height",
        "right_height",
    )
    return {key: _json_value(info[key]) for key in keys if key in info}


def _failure_reason(
    *,
    success: bool,
    task_info: dict[str, Any],
    collision: bool,
    out_of_bounds_count: int,
    safety_rejection_count: int,
    timed_out: bool,
) -> str:
    """为失败 episode 生成一个稳定且互斥的主原因。"""

    if success:
        return "success"
    if safety_rejection_count:
        return "safety_rejection"
    if collision:
        return "collision"
    if out_of_bounds_count:
        return "joint_limit_clipping"

    left_success = task_info.get("left_success")
    right_success = task_info.get("right_success")
    if left_success is False and right_success is False:
        return "left_and_right_lift_failed"
    if left_success is False:
        return "left_lift_failed"
    if right_success is False:
        return "right_lift_failed"
    if timed_out:
        return "timeout"
    if not task_info:
        return "no_task_info"
    return "task_success_condition_not_met"


def _run_episode(
    env: gym.Env,
    raw_env: Any,
    broker: action_chunk_broker.ActionChunkBroker,
    observation_adapter: GalaxeaObservationAdapter,
    action_adapter: GalaxeaActionAdapter,
    safety_filter: Any,
    *,
    seed: int,
    control_freq: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    """执行一个固定 seed 的完整闭环 episode 并收集指标。

    每个循环周期依次完成：重用当前观测构造请求、从 Broker 获取单步动作、
    做安全过滤、推进 Gym、检查任务信息和 SAPIEN 接触。Broker 自身会在
    动作块耗尽后通过 WebSocket 重新请求策略。
    """
    observation, reset_info = env.reset(seed=seed)
    broker.reset()

    # In GUI mode, create and paint the viewer before the first policy query
    # so the user can see the reset state immediately.
    if not raw_env.headless:
        env.render()

    robot_link_names = {str(link.name) for link in raw_env.robot.links}
    previous_collision_pairs: set[tuple[str, str]] = set()
    collision_pairs_seen: set[tuple[str, str]] = set()
    collision_contact_steps = 0
    collision_event_count = 0
    out_of_bounds_count = 0
    rate_limited_count = 0
    safety_rejection_count = 0
    episode_steps = 0
    done = False
    last_info: dict[str, Any] = {}

    while not done and episode_steps < max_episode_steps:
        request = observation_adapter(observation)
        policy_result = broker.infer(request)
        candidate_action = action_adapter(policy_result)
        current_state = observation_adapter.state_from_observation(observation)
        safe_action, safety_report = safety_filter.filter(candidate_action, current_state)

        if safety_report.joint_limit_clipped:
            out_of_bounds_count += 1
        if safety_report.rate_limited:
            rate_limited_count += 1
        if safety_report.rejected:
            safety_rejection_count += 1

        observation, _, terminated, truncated, last_info = env.step(safe_action)
        episode_steps += 1
        done = bool(terminated or truncated)

        if not raw_env.headless:
            env.render()

        collision_pairs = _unexpected_robot_contacts(raw_env._scene, robot_link_names)
        if collision_pairs:
            collision_contact_steps += 1
            collision_pairs_seen.update(collision_pairs)
            collision_event_count += len(collision_pairs - previous_collision_pairs)
        previous_collision_pairs = collision_pairs

    success = bool(last_info.get("success", False))
    task_info = _final_task_info(last_info)
    timed_out = not success and episode_steps >= max_episode_steps
    collision = bool(collision_pairs_seen)
    return {
        "seed": seed,
        "success": success,
        "episode_steps": episode_steps,
        "duration_seconds": episode_steps / control_freq,
        "timed_out": timed_out,
        "failure_reason": _failure_reason(
            success=success,
            task_info=task_info,
            collision=collision,
            out_of_bounds_count=out_of_bounds_count,
            safety_rejection_count=safety_rejection_count,
            timed_out=timed_out,
        ),
        "collision": collision,
        "collision_event_count": collision_event_count,
        "collision_contact_steps": collision_contact_steps,
        "collision_pairs": [list(pair) for pair in sorted(collision_pairs_seen)],
        "out_of_bounds_count": out_of_bounds_count,
        "rate_limited_count": rate_limited_count,
        "safety_rejection_count": safety_rejection_count,
        "reset_info": _json_value(reset_info),
        "task_info": task_info,
    }


def main(args: Args) -> None:
    """初始化策略客户端和 Gym 环境，运行多 seed 评测并写出 JSON。"""
    prepare_render_environment(require_display=args.gui)
    if args.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if args.execute_horizon <= 0:
        raise ValueError("execute_horizon must be positive")
    if args.execute_horizon not in (EXECUTE_ACTION_HORIZON, POLICY_ACTION_HORIZON):
        raise ValueError(
            "execute_horizon must be either "
            f"{EXECUTE_ACTION_HORIZON} (project default) or {POLICY_ACTION_HORIZON} (full chunk)"
        )
    if args.control_freq <= 0:
        raise ValueError("control_freq must be positive")

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
        api_key=args.api_key,
    )
    metadata = client.get_server_metadata()
    logging.info("Policy server metadata: %s", metadata)
    validate_galaxea_policy_metadata(metadata)
    policy = ActionChunkContractPolicy(client, action_horizon=POLICY_ACTION_HORIZON, action_dim=ACTION_DIM)
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=args.execute_horizon)

    env = gym.make(
        args.env_name,
        control_freq=args.control_freq,
        headless=not args.gui,
        obs_mode="image",
        ray_tracing=False,
        include_depth=False,
        # Match data collection and ``run_pi05_closed_loop.py``.  The
        # adapter performs the final 224px resize/pad; changing the
        # renderer scale here would change the policy's image detail
        # distribution and make the evaluation incomparable.
        camera_resolution_scale=4,
        controller_type="bimanual_joint_position",
    )
    raw_env = env.unwrapped
    registered_limit = getattr(env.spec, "max_episode_steps", None)
    max_episode_steps = args.max_episode_steps or registered_limit
    if max_episode_steps is None or max_episode_steps <= 0:
        raise ValueError("The environment has no positive episode limit; pass --max-episode-steps")

    if raw_env.action_space.shape != (ACTION_DIM,):
        raise ValueError(f"Expected a 14-D R1 action space, got {raw_env.action_space}")
    if raw_env.controller_type != "bimanual_joint_position":
        raise ValueError("The evaluator requires bimanual_joint_position")

    observation_adapter = GalaxeaObservationAdapter(default_prompt=args.prompt or args.env_name)
    action_adapter = GalaxeaActionAdapter()
    urdf_path = Path(ASSETS_DIR) / raw_env.robot.urdf_path
    safety_filter = make_r1_safety_filter(
        urdf_path,
        max_joint_delta=args.max_joint_delta if args.rate_limit else float("inf"),
        max_gripper_delta=args.max_gripper_delta if args.rate_limit else float("inf"),
    )

    seeds = list(range(args.seed_start, args.seed_start + args.num_episodes))
    episodes: list[dict[str, Any]] = []
    try:
        for index, seed in enumerate(seeds, start=1):
            print(f"Evaluating seed {seed} ({index}/{len(seeds)})", flush=True)
            result = _run_episode(
                env,
                raw_env,
                broker,
                observation_adapter,
                action_adapter,
                safety_filter,
                seed=seed,
                control_freq=args.control_freq,
                max_episode_steps=int(max_episode_steps),
            )
            episodes.append(result)
            print(
                "  "
                f"success={result['success']} "
                f"steps={result['episode_steps']} "
                f"collision={result['collision']} "
                f"out_of_bounds={result['out_of_bounds_count']}",
                flush=True,
            )
    finally:
        env.close()

    episode_count = len(episodes)
    success_count = sum(bool(item["success"]) for item in episodes)
    collision_episode_count = sum(bool(item["collision"]) for item in episodes)
    out_of_bounds_episode_count = sum(item["out_of_bounds_count"] > 0 for item in episodes)
    failure_reason_counts: dict[str, int] = {}
    for item in episodes:
        reason = str(item["failure_reason"])
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
    summary = {
        "env_name": args.env_name,
        "seeds": seeds,
        "episodes": episode_count,
        "completed_episodes": len(episodes),
        "success_count": success_count,
        "success_rate_percent": 100.0 * success_count / episode_count,
        "collision_episode_count": collision_episode_count,
        "collision_rate_percent": 100.0 * collision_episode_count / episode_count,
        "collision_event_count": sum(item["collision_event_count"] for item in episodes),
        "out_of_bounds_count": sum(item["out_of_bounds_count"] for item in episodes),
        "out_of_bounds_episode_count": out_of_bounds_episode_count,
        "failure_reason_counts": failure_reason_counts,
        "average_duration_seconds": float(np.mean([item["duration_seconds"] for item in episodes])),
        "average_episode_steps": float(np.mean([item["episode_steps"] for item in episodes])),
        "max_episode_steps": int(max_episode_steps),
        "control_freq": args.control_freq,
        "execute_horizon": args.execute_horizon,
        "rate_limit_enabled": args.rate_limit,
        "max_joint_delta": args.max_joint_delta if args.rate_limit else None,
        "max_gripper_delta": args.max_gripper_delta if args.rate_limit else None,
        "metric_definitions": {
            "success": "final task info['success'] is true",
            "collision": "an episode has an unexpected robot contact with table/wall/ground; bottle contact and wheel-ground contact are excluded",
            "out_of_bounds_count": "number of control steps whose action was clipped by the joint/gripper safety bounds",
            "failure_reason": "one per-seed primary reason; success, safety_rejection, collision, joint_limit_clipping, task-specific lift failure, timeout, or missing task info",
            "duration": "episode_steps / control_freq; simulation time, not wall-clock inference time",
        },
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
    # The simulator emits useful debug logs during setup; keep the evaluation
    # console focused on per-seed results and the final metrics.
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="WARNING")
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
