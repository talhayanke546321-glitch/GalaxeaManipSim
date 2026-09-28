"""Run the π0.5 R1 Pro policy against ROS2 in preflight, shadow or execute mode.

The default is deliberately non-actuating.  ``preflight`` only validates and
captures robot observations; ``shadow`` additionally queries the policy and
logs proposed commands.  ``execute`` creates ROS2 command publishers only
after two independent acknowledgements of the simulation-policy risk.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
from PIL import Image
import tyro

from galaxea_sim import ASSETS_DIR
from galaxea_sim.integrations.openpi.r1_pro_chunk_contract import R1ProActionChunkContractPolicy
from galaxea_sim.integrations.openpi.r1_pro_contract import (
    CONTROL_FREQUENCY,
    EXECUTE_ACTION_HORIZON,
)
from galaxea_sim.integrations.openpi.r1_pro_real_adapter import (
    R1ProHardwareActionAdapter,
    R1ProHardwareObservationAdapter,
    R1ProHardwareSnapshot,
    R1ProInitialStateReport,
    compare_r1_pro_training_initial_state,
    validate_r1_pro_real_policy_metadata,
)
from galaxea_sim.integrations.openpi.r1_pro_ros2_bridge import R1ProRos2Bridge
from galaxea_sim.integrations.openpi.r1_pro_safety import make_r1_pro_safety_filter


EXECUTION_CONFIRMATION = "R1PRO_SIM_POLICY_EXECUTION_ACKNOWLEDGED"


class Mode(enum.Enum):
    preflight = "preflight"
    shadow = "shadow"
    execute = "execute"


@dataclasses.dataclass
class Args:
    """Real-client network, observation and safety settings."""

    mode: Mode = Mode.preflight
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    api_key_env: str = "OPENPI_API_KEY"
    prompt: str = "pick up the bottle and place it on the plate"

    control_freq: float = float(CONTROL_FREQUENCY)
    execute_horizon: int = EXECUTE_ACTION_HORIZON
    max_steps: int = 300
    ros_ready_timeout_seconds: float = 30.0
    max_observation_age_seconds: float = 0.25
    max_observation_skew_seconds: float = 0.15
    max_inference_seconds: float = 2.0

    # Start far below the simulator's 0.12 rad/step limit.  These values must
    # still be reviewed against the installed robot SDK and physical setup.
    max_joint_delta: float = 0.02
    # Policy-space gripper distance per 15 Hz control step (0.05 is fully open).
    max_gripper_delta: float = 0.0025
    # Maximum velocity placed in each arm JointState command.
    arm_velocity_limit: float = 0.25
    # The expert uses only the right arm; do not replay left-arm predictions.
    hold_left_arm: bool = True
    # Execute mode requires feedback close to the common training start state.
    max_initial_joint_error: float = 0.15
    max_initial_gripper_error: float = 0.005
    # Used for joint-position clipping; replace only with the matching R1 Pro URDF.
    robot_urdf: str = str(Path(ASSETS_DIR) / "r1_pro" / "robot.urdf")

    output_dir: str = "real_robot_runs/r1_pro_bottle_place"
    allow_sim_policy_on_real_robot: bool = False
    execution_confirmation: str = ""


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _validate_args(args: Args) -> None:
    if args.control_freq != CONTROL_FREQUENCY:
        raise ValueError(f"The trained R1 Pro policy requires control_freq={CONTROL_FREQUENCY}")
    if args.execute_horizon != EXECUTE_ACTION_HORIZON:
        raise ValueError(
            f"The R1 Pro policy contract requires execute_horizon={EXECUTE_ACTION_HORIZON}"
        )
    for name in (
        "max_steps",
        "ros_ready_timeout_seconds",
        "max_observation_age_seconds",
        "max_inference_seconds",
        "max_joint_delta",
        "max_gripper_delta",
        "arm_velocity_limit",
        "max_initial_joint_error",
        "max_initial_gripper_error",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_observation_skew_seconds < 0:
        raise ValueError("max_observation_skew_seconds must be non-negative")
    if not args.prompt.strip():
        raise ValueError("prompt must not be empty")
    if args.mode is Mode.execute:
        if not args.allow_sim_policy_on_real_robot:
            raise PermissionError(
                "Execute mode uses a simulation-trained, unvalidated checkpoint. "
                "Pass --allow-sim-policy-on-real-robot only after shadow-mode review."
            )
        if args.execution_confirmation != EXECUTION_CONFIRMATION:
            raise PermissionError(
                "Execute mode requires --execution-confirmation " + EXECUTION_CONFIRMATION
            )


def _make_run_dir(root: str, mode: Mode) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root).expanduser().resolve() / f"{timestamp}-{mode.value}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _snapshot_report(
    snapshot: R1ProHardwareSnapshot,
    observation: dict,
    initial_state: R1ProInitialStateReport,
) -> dict[str, Any]:
    now = time.monotonic()
    return {
        "captured_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stream_age_seconds": {
            name: now - timestamp for name, timestamp in snapshot.receive_times.items()
        },
        "stream_skew_seconds": max(snapshot.receive_times.values())
        - min(snapshot.receive_times.values()),
        "source_image_shapes": {
            "head_rgb": list(snapshot.head_rgb.shape),
            "left_wrist_rgb": list(snapshot.left_wrist_rgb.shape),
            "right_wrist_rgb": list(snapshot.right_wrist_rgb.shape),
        },
        "policy_image_shapes": {
            name: list(image.shape) for name, image in observation["images"].items()
        },
        "policy_state": observation["state"],
        "training_initial_state": initial_state.as_dict(),
        "prompt": observation["prompt"],
    }


def _save_preflight(
    run_dir: Path,
    snapshot: R1ProHardwareSnapshot,
    observation: dict,
    initial_state: R1ProInitialStateReport,
) -> None:
    for name, image in observation["images"].items():
        Image.fromarray(image).save(run_dir / f"{name}.png")
    report = _snapshot_report(snapshot, observation, initial_state)
    (run_dir / "preflight.json").write_text(
        json.dumps(_json_value(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _hold_after_failure(bridge: R1ProRos2Bridge) -> None:
    if not bridge.enable_actuation:
        return
    try:
        bridge.publish_hold()
    except Exception:
        logging.exception("Failed to publish the emergency hold command")


def main(args: Args) -> None:
    _validate_args(args)
    execute = args.mode is Mode.execute
    run_dir = _make_run_dir(args.output_dir, args.mode)
    logging.info("R1 Pro real-client mode=%s; artifacts=%s", args.mode.value, run_dir)
    observation_adapter = R1ProHardwareObservationAdapter(
        prompt=args.prompt,
        max_age_seconds=args.max_observation_age_seconds,
        max_skew_seconds=args.max_observation_skew_seconds,
    )

    bridge = R1ProRos2Bridge(
        enable_actuation=execute,
        arm_velocity_limit=args.arm_velocity_limit,
    )
    try:
        bridge.wait_until_ready(args.ros_ready_timeout_seconds)
        snapshot = bridge.snapshot()
        observation = observation_adapter(snapshot)
        initial_state = compare_r1_pro_training_initial_state(
            observation["state"],
            max_joint_error=args.max_initial_joint_error,
            max_gripper_error=args.max_initial_gripper_error,
        )
        _save_preflight(run_dir, snapshot, observation, initial_state)
        logging.info("ROS2 preflight passed; all seven feedback streams are fresh")
        if not initial_state.within_tolerance:
            message = (
                "Robot is outside the training initial-state envelope: "
                f"max arm error={initial_state.max_arm_error:.4f} rad, "
                f"max gripper error={initial_state.max_gripper_error:.4f} policy units"
            )
            if execute:
                raise RuntimeError(message)
            logging.warning(message)
        if args.mode is Mode.preflight:
            return

        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
            api_key=api_key,
            response_timeout=args.max_inference_seconds,
        )
        metadata = client.get_server_metadata()
        validate_r1_pro_real_policy_metadata(
            metadata,
            execute=execute,
            allow_sim_policy_on_real_robot=args.allow_sim_policy_on_real_robot,
        )
        (run_dir / "policy_metadata.json").write_text(
            json.dumps(_json_value(metadata), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        policy = R1ProActionChunkContractPolicy(client)
        broker = action_chunk_broker.ActionChunkBroker(
            policy,
            action_horizon=args.execute_horizon,
        )
        safety = make_r1_pro_safety_filter(
            args.robot_urdf,
            max_joint_delta=args.max_joint_delta,
            max_gripper_delta=args.max_gripper_delta,
        )
        action_adapter = R1ProHardwareActionAdapter(
            safety,
            hold_left_arm=args.hold_left_arm,
        )
        broker.reset()
        period = 1.0 / args.control_freq
        log_path = run_dir / "steps.jsonl"
        with log_path.open("w", encoding="utf-8") as log_file:
            for step in range(args.max_steps):
                tick = time.monotonic()
                snapshot = bridge.snapshot()
                observation = observation_adapter(snapshot, now=tick)
                policy_started = time.monotonic()
                result = broker.infer(observation)
                inference_seconds = time.monotonic() - policy_started
                if inference_seconds > args.max_inference_seconds:
                    raise TimeoutError(
                        f"Policy response took {inference_seconds:.3f}s, exceeding "
                        f"the {args.max_inference_seconds:.3f}s real-robot limit"
                    )
                command, safe_action, safety_report = action_adapter(
                    result,
                    observation["state"],
                )
                if execute:
                    bridge.publish(command)

                record = {
                    "step": step,
                    "mode": args.mode.value,
                    "monotonic_time": tick,
                    "inference_seconds": inference_seconds,
                    "current_policy_state": observation["state"],
                    "raw_policy_action": result["actions"],
                    "safe_policy_action": safe_action,
                    "hardware_command": command.as_dict(),
                    "safety": safety_report.as_dict(),
                }
                log_file.write(json.dumps(_json_value(record), ensure_ascii=False) + "\n")
                log_file.flush()
                remaining = period - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        logging.warning("Interrupted by operator")
        _hold_after_failure(bridge)
    except Exception:
        _hold_after_failure(bridge)
        raise
    finally:
        bridge.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
