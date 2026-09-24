"""双臂关节位置控制器。

OpenPI 使用的外部动作顺序固定为：
``[左臂 6 关节, 左夹爪, 右臂 6 关节, 右夹爪]``。
本控制器负责把这个紧凑向量拆成环境 ``step`` 所需的四个控制信号。
"""

import gymnasium as gym
import numpy as np

from galaxea_sim.controllers.base import BaseController
from galaxea_sim.robots.bimanual import BimanualRobot
class BimanualJointPositionController(BaseController):
    """将双臂关节位置动作直接解释为 SAPIEN drive target。"""

    def __init__(self, robot: BimanualRobot, **kwargs):
        super().__init__()
        self.left_arm_indices = robot.left_arm_joint_indices
        self.right_arm_indices = robot.right_arm_joint_indices
        self.num_left_arm_joints = len(self.left_arm_indices)
        self.num_right_arm_joints = len(self.right_arm_indices)
        
        self._left_arm_target_joint_position_cmd = np.zeros(self.num_left_arm_joints, dtype=np.float32)
        self._right_arm_target_joint_position_cmd = np.zeros(self.num_right_arm_joints, dtype=np.float32)

    @property
    def left_arm_cmd(self):
        return self._left_arm_target_joint_position_cmd
    
    @property
    def right_arm_cmd(self):
        return self._right_arm_target_joint_position_cmd

    def get_control_signal(self, action):
        """拆分一个 14 维动作并返回左右臂/夹爪控制目标。

        左臂位于向量开头，左夹爪紧随其后；右臂从索引 7 开始，右夹爪
        使用最后一个元素。这里不负责限位和速度限制，那些保护逻辑位于
        OpenPI 仿真适配层的 ``ActionSafetyFilter`` 中。
        """
        self._left_arm_target_joint_position_cmd = action[:self.num_left_arm_joints]
        left_gripper_action = action[self.num_left_arm_joints]
        self._right_arm_joint_positions = action[self.num_left_arm_joints + 1:self.num_left_arm_joints + 1 + self.num_right_arm_joints]
        right_gripper_action = action[-1]
        
        return self._left_arm_target_joint_position_cmd, left_gripper_action, self._right_arm_joint_positions, right_gripper_action
    
    @property
    def action_dim(self):
        return self.num_left_arm_joints + self.num_right_arm_joints + 2
    
    @property
    def action_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32)
        
