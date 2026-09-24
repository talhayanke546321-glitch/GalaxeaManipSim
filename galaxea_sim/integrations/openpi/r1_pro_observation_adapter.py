"""把 R1 Pro SAPIEN 原始观测转换为 OpenPI 在线请求。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from galaxea_sim.utils.image_utils import prepare_rgb_image

from .r1_pro_contract import IMAGE_SIZE, pack_r1_pro_state, validate_r1_pro_vector


@dataclass(frozen=True)
class GalaxeaR1ProObservationAdapter:
    """提取三路RGB、16维关节状态和当前任务文本。"""

    image_size: tuple[int, int] = IMAGE_SIZE
    default_prompt: str = "pick up the two bottles simultaneously"

    def state_from_observation(self, observation: dict) -> np.ndarray:
        """从嵌套仿真观测中提取左右7轴关节和夹爪。"""
        upper = observation["upper_body_observations"]
        return pack_r1_pro_state(
            upper["left_arm_joint_position"],
            upper["left_arm_gripper_position"],
            upper["right_arm_joint_position"],
            upper["right_arm_gripper_position"],
        )

    def _image(self, value: object, *, name: str) -> np.ndarray:
        return prepare_rgb_image(value, *self.image_size, name=name)

    def __call__(self, observation: dict) -> dict:
        upper = observation["upper_body_observations"]
        state = validate_r1_pro_vector(self.state_from_observation(observation), name="state")
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


__all__ = ["GalaxeaR1ProObservationAdapter"]
