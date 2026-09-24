"""把 SAPIEN/Gym 原始观测转换为在线 OpenPI 请求。

仿真环境返回的是面向任务开发的嵌套字典，里面包含关节速度、末端位姿、
物体真值和动作缓存等丰富字段。策略服务不应该直接依赖这套内部结构，
所以这里定义了一个稳定的在线接口：三路 RGB、14 维状态和一条 prompt。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import IMAGE_SIZE, pack_r1_state, validate_r1_vector
from galaxea_sim.utils.image_utils import prepare_rgb_image


@dataclass(frozen=True)
class GalaxeaObservationAdapter:
    """构造 ``GalaxeaInputs`` 能够识别的紧凑观测。

    注意这里会优先采用环境提供的 ``language_instruction``。只有任务没有
    提供语言指令时，才会使用 ``default_prompt``。因此正常多任务闭环中，
    当前任务由 Gym 环境本身的语言指令传给模型，而不是由图像识别猜测。
    """

    image_size: tuple[int, int] = IMAGE_SIZE
    default_prompt: str = "pick up the two bottles simultaneously"

    def state_from_observation(self, observation: dict) -> np.ndarray:
        """从原始观测提取左右臂关节和夹爪，拼成 14 维状态。"""
        upper = observation["upper_body_observations"]
        return pack_r1_state(
            upper["left_arm_joint_position"],
            upper["left_arm_gripper_position"],
            upper["right_arm_joint_position"],
            upper["right_arm_gripper_position"],
        )

    def _image(self, value: object, *, name: str) -> np.ndarray:
        """把相机图像转成连续的 HWC、uint8、224x224 RGB 数组。"""
        return prepare_rgb_image(value, *self.image_size, name=name)

    def __call__(self, observation: dict) -> dict:
        """执行一次原始仿真观测到在线请求的字段映射。"""
        upper = observation["upper_body_observations"]
        state = validate_r1_vector(self.state_from_observation(observation), name="state")
        prompt = observation.get("language_instruction") or self.default_prompt
        return {
            "images": {
                "cam_high": self._image(upper["rgb_head"], name="rgb_head"),
                "cam_left_wrist": self._image(upper["rgb_left_hand"], name="rgb_left_hand"),
                "cam_right_wrist": self._image(upper["rgb_right_hand"], name="rgb_right_hand"),
            },
            "state": state,
            "prompt": str(prompt),
        }


__all__ = ["GalaxeaObservationAdapter"]
