"""Galaxea R1 Pro 仿真器与 OpenPI 之间的16维在线契约。

R1 Pro 每只手臂有7个关节。状态和动作严格按
``左臂7 + 左夹爪1 + 右臂7 + 右夹爪1`` 排列；仿真夹爪单位是米，0闭合、
0.05张开。协议与标准 R1 的14维契约分开定义，防止形状相近时误接策略。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ARM_DOF = 7
GRIPPER_MAX = 0.05
ACTION_DIM = 2 * (ARM_DOF + 1)
IMAGE_SIZE = (224, 224)
CONTROL_FREQUENCY = 15
POLICY_ACTION_HORIZON = 15
EXECUTE_ACTION_HORIZON = 10

LEFT_ARM_JOINT_NAMES = tuple(f"left_arm_joint{i}" for i in range(1, ARM_DOF + 1))
RIGHT_ARM_JOINT_NAMES = tuple(f"right_arm_joint{i}" for i in range(1, ARM_DOF + 1))
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
STATE_ORDER = (
    *LEFT_ARM_JOINT_NAMES,
    "left_gripper",
    *RIGHT_ARM_JOINT_NAMES,
    "right_gripper",
)


def pack_r1_pro_state(
    left_arm_joints: Sequence[float],
    left_gripper: Sequence[float] | float,
    right_arm_joints: Sequence[float],
    right_gripper: Sequence[float] | float,
) -> np.ndarray:
    """按固定顺序拼接两条7轴手臂和两个夹爪。"""
    left_arm = np.asarray(left_arm_joints, dtype=np.float32).reshape(-1)
    right_arm = np.asarray(right_arm_joints, dtype=np.float32).reshape(-1)
    left_grip = np.asarray(left_gripper, dtype=np.float32).reshape(-1)
    right_grip = np.asarray(right_gripper, dtype=np.float32).reshape(-1)
    if left_arm.size != ARM_DOF or right_arm.size != ARM_DOF:
        raise ValueError(f"R1 Pro arm state must have {ARM_DOF} joints per arm")
    if left_grip.size != 1 or right_grip.size != 1:
        raise ValueError("R1 Pro gripper state must contain one scalar per arm")
    return np.concatenate([left_arm, left_grip, right_arm, right_grip]).astype(np.float32, copy=False)


def validate_r1_pro_vector(value: object, *, name: str = "value") -> np.ndarray:
    """校验16维向量形状和有限值。"""
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


__all__ = [
    "ACTION_DIM",
    "ARM_DOF",
    "ARM_JOINT_NAMES",
    "CONTROL_FREQUENCY",
    "EXECUTE_ACTION_HORIZON",
    "GRIPPER_MAX",
    "IMAGE_SIZE",
    "POLICY_ACTION_HORIZON",
    "STATE_ORDER",
    "pack_r1_pro_state",
    "validate_r1_pro_vector",
]
