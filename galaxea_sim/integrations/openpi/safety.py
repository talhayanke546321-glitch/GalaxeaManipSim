"""R1 仿真闭环的最后一道动作安全层。

策略输出不是直接的物理控制命令。本文件先检查其是否可解析且为有限值，
再根据 R1 URDF 的关节上下限做硬裁剪，最后限制相对于当前状态的单步变化
量。无法安全解释的动作会被拒绝，并保持机器人当前状态不变。
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .contract import ACTION_DIM, ARM_JOINT_NAMES, GRIPPER_MAX


@dataclasses.dataclass(frozen=True)
class SafetyLimits:
    """每个动作维度的下限、上限和最大单步变化量。"""

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
            raise ValueError("invalid safety limits")


@dataclasses.dataclass(frozen=True)
class SafetyReport:
    """记录一次动作是否被拒绝、裁剪或限速。"""

    rejected: bool = False
    nonfinite: bool = False
    joint_limit_clipped: bool = False
    rate_limited: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class ActionSafetyFilter:
    """对原始动作应用硬边界和每步 slew-rate 限制。"""

    def __init__(self, limits: SafetyLimits):
        self.limits = limits

    def filter(self, action: object, current_state: object) -> tuple[np.ndarray, SafetyReport]:
        """返回安全动作及诊断报告。

        处理顺序是：

        1. 验证当前状态是有限的 14 维向量；
        2. 验证候选动作可转成有限的 14 维向量；
        3. 按 URDF/夹爪范围做硬裁剪；
        4. 按最大单步变化量限速；
        5. 再次裁剪到物理范围后返回。
        """
        current = np.asarray(current_state, dtype=np.float32)
        if current.shape != (ACTION_DIM,) or not np.all(np.isfinite(current)):
            raise ValueError("current R1 state must be a finite 14-dimensional vector")

        try:
            candidate = np.asarray(action, dtype=np.float32)
        except (TypeError, ValueError):
            return current.copy(), SafetyReport(rejected=True, reason="action is not numeric")

        if candidate.shape != (ACTION_DIM,):
            return current.copy(), SafetyReport(
                rejected=True,
                reason=f"action shape is {candidate.shape}, expected ({ACTION_DIM},)",
            )
        if not np.all(np.isfinite(candidate)):
            return current.copy(), SafetyReport(rejected=True, nonfinite=True, reason="action contains NaN/Inf")

        bounded = np.clip(candidate, self.limits.lower, self.limits.upper)
        joint_limit_clipped = not np.array_equal(bounded, candidate)

        delta = bounded - current
        limited = current + np.clip(delta, -self.limits.max_delta, self.limits.max_delta)
        limited = np.clip(limited, self.limits.lower, self.limits.upper)
        rate_limited = not np.array_equal(limited, bounded)
        return limited.astype(np.float32, copy=False), SafetyReport(
            joint_limit_clipped=joint_limit_clipped,
            rate_limited=rate_limited,
        )


def load_joint_limits(urdf_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 URDF 中左右臂关节限位，并按策略动作顺序排列。"""

    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is not None and upper is not None:
            limits[name] = (float(lower), float(upper))

    missing = [name for name in ARM_JOINT_NAMES if name not in limits]
    if missing:
        raise ValueError(f"URDF has no limits for R1 arm joints: {missing}")

    arm_lower = np.asarray([limits[name][0] for name in ARM_JOINT_NAMES], dtype=np.float32)
    arm_upper = np.asarray([limits[name][1] for name in ARM_JOINT_NAMES], dtype=np.float32)
    return arm_lower, arm_upper


def make_r1_safety_filter(
    urdf_path: str | Path,
    *,
    max_joint_delta: float = 0.12,
    max_gripper_delta: float = 0.01,
) -> ActionSafetyFilter:
    """根据 R1 URDF 创建包含夹爪范围的安全过滤器。"""
    arm_lower, arm_upper = load_joint_limits(urdf_path)
    lower = np.concatenate([arm_lower[:6], [0.0], arm_lower[6:], [0.0]]).astype(np.float32)
    upper = np.concatenate([arm_upper[:6], [GRIPPER_MAX], arm_upper[6:], [GRIPPER_MAX]]).astype(np.float32)
    max_delta = np.concatenate(
        [
            np.full(6, max_joint_delta, dtype=np.float32),
            [max_gripper_delta],
            np.full(6, max_joint_delta, dtype=np.float32),
            [max_gripper_delta],
        ]
    )
    return ActionSafetyFilter(SafetyLimits(lower=lower, upper=upper, max_delta=max_delta))


__all__ = [
    "ActionSafetyFilter",
    "SafetyLimits",
    "SafetyReport",
    "load_joint_limits",
    "make_r1_safety_filter",
]
