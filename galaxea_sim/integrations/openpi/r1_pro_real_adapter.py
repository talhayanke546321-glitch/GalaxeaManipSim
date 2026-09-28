"""Pure-data adapters between the OpenPI R1 Pro contract and real hardware.

This module deliberately has no ROS2 dependency.  It owns the unit boundary
that is easy to get wrong during sim-to-real deployment:

* OpenPI uses a flat 16-D vector ordered as left arm 7, left gripper, right
  arm 7, right gripper.
* Arm positions are radians on both sides.
* The simulator-facing policy uses gripper values in ``[0.0, 0.05]`` while
  the R1 Pro motion controller uses a stroke in ``[0, 100]``.
* Camera inputs are RGB HWC uint8 images resized with the same pad contract as
  the training data.

Keeping these transforms separate from ``rclpy`` lets us test every numerical
boundary on the GPU workstation before connecting a robot.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import time

import numpy as np

from galaxea_sim.utils.image_utils import prepare_rgb_image

from .r1_pro_action_adapter import GalaxeaR1ProActionAdapter
from .r1_pro_chunk_contract import validate_r1_pro_policy_metadata
from .r1_pro_contract import (
    ARM_DOF,
    CONTROL_FREQUENCY,
    GRIPPER_MAX,
    IMAGE_SIZE,
    STATE_ORDER,
    pack_r1_pro_state,
)
from .r1_pro_safety import R1ProActionSafetyFilter, R1ProSafetyReport


HARDWARE_GRIPPER_CLOSED = 0.0
HARDWARE_GRIPPER_OPEN = 100.0
# Every one of the 120 demonstrations in ``bottle_place_high_v1`` starts from
# this exact state.  The policy predicts absolute joint targets, so powered
# execution from an unrelated pose is an avoidable sim-to-real hazard.
TRAINING_INITIAL_POLICY_STATE = np.asarray(
    [
        -0.4,
        1.3,
        -0.7,
        -1.57,
        1.3,
        -0.4,
        -0.8,
        0.0,
        -0.4,
        -1.3,
        0.7,
        -1.57,
        -1.3,
        -0.4,
        0.8,
        0.0,
    ],
    dtype=np.float32,
)
_ARM_STATE_INDICES = np.asarray([*range(7), *range(8, 15)])
_GRIPPER_STATE_INDICES = np.asarray([7, 15])
SNAPSHOT_KEYS = (
    "head_rgb",
    "left_wrist_rgb",
    "right_wrist_rgb",
    "left_arm",
    "left_gripper",
    "right_arm",
    "right_gripper",
)


def hardware_gripper_to_policy(value: object) -> float:
    """Convert the R1 Pro SDK's 0..100 stroke to the policy's 0..0.05 value."""
    scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError("hardware gripper feedback must be finite")
    tolerance = 1e-5
    if not HARDWARE_GRIPPER_CLOSED - tolerance <= scalar <= HARDWARE_GRIPPER_OPEN + tolerance:
        raise ValueError(f"hardware gripper feedback must be in [0, 100], got {scalar}")
    scalar = float(np.clip(scalar, HARDWARE_GRIPPER_CLOSED, HARDWARE_GRIPPER_OPEN))
    return scalar / HARDWARE_GRIPPER_OPEN * GRIPPER_MAX


def policy_gripper_to_hardware(value: object) -> float:
    """Convert the policy's 0..0.05 value to the R1 Pro SDK's 0..100 stroke."""
    scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError("policy gripper command must be finite")
    tolerance = 1e-6
    if not -tolerance <= scalar <= GRIPPER_MAX + tolerance:
        raise ValueError(f"policy gripper command must be in [0, {GRIPPER_MAX}], got {scalar}")
    scalar = float(np.clip(scalar, 0.0, GRIPPER_MAX))
    return scalar / GRIPPER_MAX * HARDWARE_GRIPPER_OPEN


def _finite_vector(value: object, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclasses.dataclass(frozen=True)
class R1ProHardwareSnapshot:
    """One synchronized-enough set of R1 Pro camera and joint observations."""

    head_rgb: np.ndarray
    left_wrist_rgb: np.ndarray
    right_wrist_rgb: np.ndarray
    left_arm: np.ndarray
    left_gripper: np.ndarray
    right_arm: np.ndarray
    right_gripper: np.ndarray
    receive_times: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
            image = np.asarray(getattr(self, name))
            if image.ndim != 3 or image.shape[-1] not in (3, 4):
                raise ValueError(f"{name} must be an HWC RGB/RGBA image, got {image.shape}")
            object.__setattr__(self, name, image)
        object.__setattr__(self, "left_arm", _finite_vector(self.left_arm, name="left_arm", size=ARM_DOF))
        object.__setattr__(self, "right_arm", _finite_vector(self.right_arm, name="right_arm", size=ARM_DOF))
        object.__setattr__(
            self,
            "left_gripper",
            _finite_vector(self.left_gripper, name="left_gripper", size=1),
        )
        object.__setattr__(
            self,
            "right_gripper",
            _finite_vector(self.right_gripper, name="right_gripper", size=1),
        )
        missing = set(SNAPSHOT_KEYS) - set(self.receive_times)
        if missing:
            raise ValueError(f"snapshot is missing receive timestamps: {sorted(missing)}")
        receive_times = {name: float(self.receive_times[name]) for name in SNAPSHOT_KEYS}
        if not all(np.isfinite(value) for value in receive_times.values()):
            raise ValueError("snapshot receive timestamps must be finite")
        object.__setattr__(self, "receive_times", receive_times)

    def validate_timing(
        self,
        *,
        now: float | None = None,
        max_age_seconds: float,
        max_skew_seconds: float,
    ) -> None:
        """Reject stale observations or a camera/state set with excessive skew."""
        if max_age_seconds <= 0 or max_skew_seconds < 0:
            raise ValueError("observation age must be positive and skew must be non-negative")
        current = time.monotonic() if now is None else float(now)
        newest = max(self.receive_times.values())
        oldest = min(self.receive_times.values())
        age = current - newest
        skew = newest - oldest
        if age < -1e-3:
            raise ValueError(f"snapshot timestamp is {abs(age):.3f}s in the future")
        if age > max_age_seconds:
            raise TimeoutError(
                f"R1 Pro observation is stale by {age:.3f}s (limit {max_age_seconds:.3f}s)"
            )
        if skew > max_skew_seconds:
            raise TimeoutError(
                f"R1 Pro observation skew is {skew:.3f}s (limit {max_skew_seconds:.3f}s)"
            )

    def policy_state(self) -> np.ndarray:
        """Return the model-facing 16-D absolute position state."""
        return pack_r1_pro_state(
            self.left_arm,
            hardware_gripper_to_policy(self.left_gripper),
            self.right_arm,
            hardware_gripper_to_policy(self.right_gripper),
        )


@dataclasses.dataclass(frozen=True)
class R1ProHardwareCommand:
    """One command expressed in the units accepted by the R1 Pro ROS2 SDK."""

    left_arm: np.ndarray
    left_gripper: float
    right_arm: np.ndarray
    right_gripper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_arm", _finite_vector(self.left_arm, name="left_arm", size=ARM_DOF))
        object.__setattr__(self, "right_arm", _finite_vector(self.right_arm, name="right_arm", size=ARM_DOF))
        for name in ("left_gripper", "right_gripper"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not HARDWARE_GRIPPER_CLOSED <= value <= HARDWARE_GRIPPER_OPEN:
                raise ValueError(f"{name} must be a finite hardware stroke in [0, 100]")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, object]:
        return {
            "left_arm": self.left_arm.tolist(),
            "left_gripper": self.left_gripper,
            "right_arm": self.right_arm.tolist(),
            "right_gripper": self.right_gripper,
        }


@dataclasses.dataclass(frozen=True)
class R1ProInitialStateReport:
    """Distance between live feedback and the demonstrations' initial state."""

    current: np.ndarray
    expected: np.ndarray
    absolute_error: np.ndarray
    max_arm_error: float
    max_gripper_error: float
    within_tolerance: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "current": self.current.tolist(),
            "expected": self.expected.tolist(),
            "absolute_error": self.absolute_error.tolist(),
            "max_arm_error": self.max_arm_error,
            "max_gripper_error": self.max_gripper_error,
            "within_tolerance": self.within_tolerance,
        }


def compare_r1_pro_training_initial_state(
    current_policy_state: object,
    *,
    max_joint_error: float = 0.15,
    max_gripper_error: float = 0.005,
) -> R1ProInitialStateReport:
    """Compare live state to the common initial state of all training demos."""
    if max_joint_error <= 0 or max_gripper_error <= 0:
        raise ValueError("initial-state tolerances must be positive")
    current = _finite_vector(
        current_policy_state,
        name="current_policy_state",
        size=len(STATE_ORDER),
    )
    expected = TRAINING_INITIAL_POLICY_STATE.copy()
    error = np.abs(current - expected)
    max_arm_error_seen = float(np.max(error[_ARM_STATE_INDICES]))
    max_gripper_error_seen = float(np.max(error[_GRIPPER_STATE_INDICES]))
    return R1ProInitialStateReport(
        current=current.copy(),
        expected=expected,
        absolute_error=error,
        max_arm_error=max_arm_error_seen,
        max_gripper_error=max_gripper_error_seen,
        within_tolerance=(
            max_arm_error_seen <= max_joint_error
            and max_gripper_error_seen <= max_gripper_error
        ),
    )


@dataclasses.dataclass(frozen=True)
class R1ProHardwareObservationAdapter:
    """Build the exact OpenPI observation used by the simulation-trained policy."""

    prompt: str = "pick up the bottle and place it on the plate"
    image_size: tuple[int, int] = IMAGE_SIZE
    max_age_seconds: float = 0.25
    max_skew_seconds: float = 0.15

    def __call__(self, snapshot: R1ProHardwareSnapshot, *, now: float | None = None) -> dict:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        snapshot.validate_timing(
            now=now,
            max_age_seconds=self.max_age_seconds,
            max_skew_seconds=self.max_skew_seconds,
        )
        height, width = self.image_size
        return {
            "images": {
                "cam_high": prepare_rgb_image(snapshot.head_rgb, height, width, name="head_rgb"),
                "cam_left_wrist": prepare_rgb_image(
                    snapshot.left_wrist_rgb, height, width, name="left_wrist_rgb"
                ),
                "cam_right_wrist": prepare_rgb_image(
                    snapshot.right_wrist_rgb, height, width, name="right_wrist_rgb"
                ),
            },
            "state": snapshot.policy_state(),
            "prompt": self.prompt,
        }


class R1ProHardwareActionAdapter:
    """Validate, limit and convert one OpenPI action before ROS publication."""

    def __init__(self, safety: R1ProActionSafetyFilter, *, hold_left_arm: bool = True) -> None:
        self._policy_action = GalaxeaR1ProActionAdapter()
        self._safety = safety
        self._hold_left_arm = hold_left_arm

    def __call__(
        self,
        policy_result: dict | np.ndarray,
        current_policy_state: object,
    ) -> tuple[R1ProHardwareCommand, np.ndarray, R1ProSafetyReport]:
        current = _finite_vector(current_policy_state, name="current_policy_state", size=len(STATE_ORDER))
        candidate = self._policy_action(policy_result)
        safe, report = self._safety.filter(candidate, current)
        if self._hold_left_arm:
            safe[: ARM_DOF + 1] = current[: ARM_DOF + 1]
        command = R1ProHardwareCommand(
            left_arm=safe[:ARM_DOF],
            left_gripper=policy_gripper_to_hardware(safe[ARM_DOF]),
            right_arm=safe[ARM_DOF + 1 : -1],
            right_gripper=policy_gripper_to_hardware(safe[-1]),
        )
        return command, safe, report


def validate_r1_pro_real_policy_metadata(
    metadata: Mapping[str, object],
    *,
    execute: bool,
    allow_sim_policy_on_real_robot: bool,
) -> None:
    """Validate the wire contract and fail closed for unacknowledged real execution."""
    validate_r1_pro_policy_metadata(metadata)
    expected = {
        "control_frequency_hz": CONTROL_FREQUENCY,
        "image_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "arm_position_unit": "radian",
        "image_layout": "HWC",
        "image_dtype": "uint8",
        "image_color_space": "RGB",
    }
    missing = [key for key in expected if key not in metadata]
    if missing:
        raise ValueError(f"Policy server metadata is missing real-client fields: {missing}")
    mismatches = {key: (metadata[key], value) for key, value in expected.items() if metadata[key] != value}
    if mismatches:
        raise ValueError(f"Policy server metadata is incompatible with the real client: {mismatches}")
    if execute and not bool(metadata.get("real_robot_validated", False)) and not allow_sim_policy_on_real_robot:
        raise PermissionError(
            "Policy metadata says real_robot_validated=false. Run shadow mode first; "
            "real execution additionally requires --allow-sim-policy-on-real-robot."
        )


__all__ = [
    "HARDWARE_GRIPPER_CLOSED",
    "HARDWARE_GRIPPER_OPEN",
    "R1ProHardwareActionAdapter",
    "R1ProHardwareCommand",
    "R1ProInitialStateReport",
    "R1ProHardwareObservationAdapter",
    "R1ProHardwareSnapshot",
    "SNAPSHOT_KEYS",
    "TRAINING_INITIAL_POLICY_STATE",
    "compare_r1_pro_training_initial_state",
    "hardware_gripper_to_policy",
    "policy_gripper_to_hardware",
    "validate_r1_pro_real_policy_metadata",
]
