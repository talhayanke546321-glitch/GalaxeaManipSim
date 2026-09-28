import time

import numpy as np
import pytest

from galaxea_sim.integrations.openpi.r1_pro_real_adapter import (
    R1ProHardwareActionAdapter,
    R1ProHardwareObservationAdapter,
    R1ProHardwareSnapshot,
    TRAINING_INITIAL_POLICY_STATE,
    compare_r1_pro_training_initial_state,
    hardware_gripper_to_policy,
    policy_gripper_to_hardware,
    validate_r1_pro_real_policy_metadata,
)
from galaxea_sim.integrations.openpi.r1_pro_safety import (
    R1ProActionSafetyFilter,
    R1ProSafetyLimits,
)


def _snapshot(*, now: float | None = None, skew: float = 0.0) -> R1ProHardwareSnapshot:
    timestamp = time.monotonic() if now is None else now
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    receive_times = {
        "head_rgb": timestamp,
        "left_wrist_rgb": timestamp - skew,
        "right_wrist_rgb": timestamp,
        "left_arm": timestamp,
        "left_gripper": timestamp,
        "right_arm": timestamp,
        "right_gripper": timestamp,
    }
    return R1ProHardwareSnapshot(
        head_rgb=image,
        left_wrist_rgb=image,
        right_wrist_rgb=image,
        left_arm=np.arange(7, dtype=np.float32) / 10,
        left_gripper=np.asarray([100.0], dtype=np.float32),
        right_arm=-np.arange(7, dtype=np.float32) / 10,
        right_gripper=np.asarray([0.0], dtype=np.float32),
        receive_times=receive_times,
    )


def _metadata(**updates) -> dict:
    metadata = {
        "protocol_version": "galaxea-r1-pro-openpi-v1",
        "robot": "r1_pro",
        "controller": "bimanual_joint_position",
        "action_dim": 16,
        "action_horizon": 15,
        "execute_horizon": 10,
        "control_frequency_hz": 15,
        "state_order": [
            *(f"left_arm_joint{i}" for i in range(1, 8)),
            "left_gripper",
            *(f"right_arm_joint{i}" for i in range(1, 8)),
            "right_gripper",
        ],
        "arm_position_unit": "radian",
        "image_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "image_layout": "HWC",
        "image_dtype": "uint8",
        "image_color_space": "RGB",
        "training_domain": "simulation",
        "real_robot_validated": False,
    }
    metadata.update(updates)
    return metadata


def test_real_gripper_unit_conversion_round_trip() -> None:
    assert hardware_gripper_to_policy(0.0) == 0.0
    assert hardware_gripper_to_policy(100.0) == pytest.approx(0.05)
    assert policy_gripper_to_hardware(0.0) == 0.0
    assert policy_gripper_to_hardware(0.05) == pytest.approx(100.0)
    for hardware_value in (0.0, 25.0, 50.0, 100.0):
        assert policy_gripper_to_hardware(hardware_gripper_to_policy(hardware_value)) == pytest.approx(
            hardware_value
        )
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        hardware_gripper_to_policy(101.0)
    with pytest.raises(ValueError, match=r"\[0, 0.05\]"):
        policy_gripper_to_hardware(-0.01)


def test_real_observation_matches_training_contract() -> None:
    now = time.monotonic()
    observation = R1ProHardwareObservationAdapter()(_snapshot(now=now), now=now)
    assert observation["state"].shape == (16,)
    np.testing.assert_allclose(observation["state"][[7, 15]], [0.05, 0.0])
    assert set(observation["images"]) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    assert all(image.shape == (224, 224, 3) for image in observation["images"].values())
    assert all(image.dtype == np.uint8 for image in observation["images"].values())


def test_real_observation_rejects_stale_or_skewed_streams() -> None:
    adapter = R1ProHardwareObservationAdapter(max_age_seconds=0.2, max_skew_seconds=0.1)
    now = time.monotonic()
    with pytest.raises(TimeoutError, match="stale"):
        adapter(_snapshot(now=now - 1.0), now=now)
    with pytest.raises(TimeoutError, match="skew"):
        adapter(_snapshot(now=now, skew=0.2), now=now)


def test_real_action_is_rate_limited_and_left_side_held() -> None:
    lower = np.full(16, -2.0, dtype=np.float32)
    upper = np.full(16, 2.0, dtype=np.float32)
    lower[[7, 15]] = 0.0
    upper[[7, 15]] = 0.05
    safety = R1ProActionSafetyFilter(
        R1ProSafetyLimits(
            lower=lower,
            upper=upper,
            max_delta=np.asarray([0.02] * 7 + [0.0025] + [0.02] * 7 + [0.0025]),
        )
    )
    adapter = R1ProHardwareActionAdapter(safety, hold_left_arm=True)
    current = np.zeros(16, dtype=np.float32)
    candidate = np.ones(16, dtype=np.float32)
    command, safe, report = adapter({"actions": candidate}, current)
    assert report.joint_limit_clipped
    assert report.rate_limited
    np.testing.assert_array_equal(safe[:8], current[:8])
    np.testing.assert_allclose(safe[8:15], 0.02)
    assert command.left_gripper == 0.0
    assert command.right_gripper == pytest.approx(5.0)


def test_real_execution_requires_explicit_sim_policy_acknowledgement() -> None:
    validate_r1_pro_real_policy_metadata(
        _metadata(),
        execute=False,
        allow_sim_policy_on_real_robot=False,
    )
    with pytest.raises(PermissionError, match="real_robot_validated=false"):
        validate_r1_pro_real_policy_metadata(
            _metadata(),
            execute=True,
            allow_sim_policy_on_real_robot=False,
        )
    validate_r1_pro_real_policy_metadata(
        _metadata(),
        execute=True,
        allow_sim_policy_on_real_robot=True,
    )


def test_training_initial_state_report_gates_powered_execution_pose() -> None:
    matching = compare_r1_pro_training_initial_state(TRAINING_INITIAL_POLICY_STATE)
    assert matching.within_tolerance
    assert matching.max_arm_error == 0.0
    assert matching.max_gripper_error == 0.0

    displaced = TRAINING_INITIAL_POLICY_STATE.copy()
    displaced[8] += 0.2
    displaced[15] += 0.006
    report = compare_r1_pro_training_initial_state(
        displaced,
        max_joint_error=0.15,
        max_gripper_error=0.005,
    )
    assert not report.within_tolerance
    assert report.max_arm_error == pytest.approx(0.2)
    assert report.max_gripper_error == pytest.approx(0.006)
