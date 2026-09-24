"""R1 Pro 仿真闭环的16维关节限位和单步限速层。"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .r1_pro_contract import ACTION_DIM, ARM_DOF, ARM_JOINT_NAMES, GRIPPER_MAX


@dataclasses.dataclass(frozen=True)
class R1ProSafetyLimits:
    lower: np.ndarray
    upper: np.ndarray
    max_delta: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (("lower", self.lower), ("upper", self.upper), ("max_delta", self.max_delta)):
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (ACTION_DIM,):
                raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {array.shape}")
            object.__setattr__(self, name, array)
        if np.any(self.lower > self.upper) or np.any(self.max_delta <= 0):
            raise ValueError("invalid R1 Pro safety limits")


@dataclasses.dataclass(frozen=True)
class R1ProSafetyReport:
    rejected: bool = False
    nonfinite: bool = False
    joint_limit_clipped: bool = False
    rate_limited: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class R1ProActionSafetyFilter:
    """异常动作保持当前位姿，正常动作依次经过限位和限速。"""

    def __init__(self, limits: R1ProSafetyLimits):
        self.limits = limits

    def filter(self, action: object, current_state: object) -> tuple[np.ndarray, R1ProSafetyReport]:
        current = np.asarray(current_state, dtype=np.float32)
        if current.shape != (ACTION_DIM,) or not np.all(np.isfinite(current)):
            raise ValueError("current R1 Pro state must be a finite 16-dimensional vector")
        try:
            candidate = np.asarray(action, dtype=np.float32)
        except (TypeError, ValueError):
            return current.copy(), R1ProSafetyReport(rejected=True, reason="action is not numeric")
        if candidate.shape != (ACTION_DIM,):
            return current.copy(), R1ProSafetyReport(
                rejected=True,
                reason=f"action shape is {candidate.shape}, expected ({ACTION_DIM},)",
            )
        if not np.all(np.isfinite(candidate)):
            return current.copy(), R1ProSafetyReport(
                rejected=True,
                nonfinite=True,
                reason="action contains NaN/Inf",
            )

        bounded = np.clip(candidate, self.limits.lower, self.limits.upper)
        joint_limit_clipped = not np.array_equal(bounded, candidate)
        limited = current + np.clip(
            bounded - current,
            -self.limits.max_delta,
            self.limits.max_delta,
        )
        limited = np.clip(limited, self.limits.lower, self.limits.upper)
        return limited.astype(np.float32, copy=False), R1ProSafetyReport(
            joint_limit_clipped=joint_limit_clipped,
            rate_limited=not np.array_equal(limited, bounded),
        )


def load_r1_pro_joint_limits(urdf_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """按左7轴、右7轴顺序读取 R1 Pro URDF 关节限位。"""
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        lower, upper = limit.get("lower"), limit.get("upper")
        if lower is not None and upper is not None:
            limits[name] = (float(lower), float(upper))
    missing = [name for name in ARM_JOINT_NAMES if name not in limits]
    if missing:
        raise ValueError(f"URDF has no limits for R1 Pro arm joints: {missing}")
    return (
        np.asarray([limits[name][0] for name in ARM_JOINT_NAMES], dtype=np.float32),
        np.asarray([limits[name][1] for name in ARM_JOINT_NAMES], dtype=np.float32),
    )


def make_r1_pro_safety_filter(
    urdf_path: str | Path,
    *,
    max_joint_delta: float = 0.12,
    max_gripper_delta: float = 0.01,
) -> R1ProActionSafetyFilter:
    """根据14个手臂关节限位构造16维安全边界。"""
    if max_joint_delta <= 0 or max_gripper_delta <= 0:
        raise ValueError("R1 Pro max deltas must be positive")
    arm_lower, arm_upper = load_r1_pro_joint_limits(urdf_path)
    lower = np.concatenate(
        [arm_lower[:ARM_DOF], [0.0], arm_lower[ARM_DOF:], [0.0]]
    ).astype(np.float32)
    upper = np.concatenate(
        [arm_upper[:ARM_DOF], [GRIPPER_MAX], arm_upper[ARM_DOF:], [GRIPPER_MAX]]
    ).astype(np.float32)
    max_delta = np.concatenate(
        [
            np.full(ARM_DOF, max_joint_delta, dtype=np.float32),
            [max_gripper_delta],
            np.full(ARM_DOF, max_joint_delta, dtype=np.float32),
            [max_gripper_delta],
        ]
    ).astype(np.float32)
    return R1ProActionSafetyFilter(R1ProSafetyLimits(lower, upper, max_delta))


__all__ = [
    "R1ProActionSafetyFilter",
    "R1ProSafetyLimits",
    "R1ProSafetyReport",
    "load_r1_pro_joint_limits",
    "make_r1_pro_safety_filter",
]
