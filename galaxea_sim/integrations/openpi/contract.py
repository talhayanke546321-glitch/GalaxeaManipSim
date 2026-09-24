"""标准 Galaxea R1 与 OpenPI 之间的唯一数据契约。

本文件故意不依赖 Gym、SAPIEN 或具体策略模型，只保存双方必须一致的
维度、顺序、单位和时间参数。仿真端、OpenPI transform、数据转换器和
安全过滤器都应该引用这里的常量，避免在多个文件里分别写出不同的
“14 维动作”定义。

当前闭环协议为：

* 每只手臂 6 个关节，共 12 个关节；
* 每只手臂 1 个夹爪标量，共 14 维动作/状态；
* 模型一次预测 15 步，客户端只执行前 10 步后重新观测；
* 控制频率为 15 Hz，图像输入统一为 224x224 RGB。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ARM_DOF = 6
GRIPPER_MAX = 0.05
ACTION_DIM = 2 * (ARM_DOF + 1)
IMAGE_SIZE = (224, 224)
CONTROL_FREQUENCY = 15
# 模型输出 15 步动作块；闭环客户端只消费前 10 步，然后重新读取图像和
# 状态并请求新的动作块。这两个数字分别对应“预测长度”和“执行长度”，
# 不能与 π0.5 内部的 flow-matching 推理迭代次数混淆。
POLICY_ACTION_HORIZON = 15
EXECUTE_ACTION_HORIZON = 10

LEFT_ARM_JOINT_NAMES = tuple(f"left_arm_joint{i}" for i in range(1, ARM_DOF + 1))
RIGHT_ARM_JOINT_NAMES = tuple(f"right_arm_joint{i}" for i in range(1, ARM_DOF + 1))
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
ACTION_JOINT_NAMES = (
    *LEFT_ARM_JOINT_NAMES,
    "left_gripper",
    *RIGHT_ARM_JOINT_NAMES,
    "right_gripper",
)


def pack_r1_state(
    left_arm_joints: Sequence[float],
    left_gripper: Sequence[float] | float,
    right_arm_joints: Sequence[float],
    right_gripper: Sequence[float] | float,
) -> np.ndarray:
    """按固定顺序拼接仿真器中的四类字段。

    输入的左右臂关节可以是 list、NumPy 数组或其它序列；夹爪可以是
    标量，也可以是长度为 1 的数组。函数会把它们统一为 float32，并
    严格检查每只手臂恰好 6 个关节、每个夹爪恰好 1 个标量。
    """

    left_arm = np.asarray(left_arm_joints, dtype=np.float32).reshape(-1)
    right_arm = np.asarray(right_arm_joints, dtype=np.float32).reshape(-1)
    left_grip = np.asarray(left_gripper, dtype=np.float32).reshape(-1)
    right_grip = np.asarray(right_gripper, dtype=np.float32).reshape(-1)
    if left_arm.size != ARM_DOF or right_arm.size != ARM_DOF:
        raise ValueError(f"R1 arm state must have {ARM_DOF} joints per arm")
    if left_grip.size != 1 or right_grip.size != 1:
        raise ValueError("R1 gripper state must contain one scalar per arm")
    return np.concatenate([left_arm, left_grip, right_arm, right_grip]).astype(np.float32, copy=False)


def validate_r1_vector(value: object, *, name: str = "value") -> np.ndarray:
    """检查一个原始 R1 状态/动作向量是否可以进入闭环。

    这里的检查只负责形状和有限值；关节物理限位及每步最大变化量由
    ``safety.py`` 中的安全过滤器负责。分层检查可以更早定位是协议错误
    还是物理安全约束触发。
    """

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
