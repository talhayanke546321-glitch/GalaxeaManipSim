"""双臂操作环境的公共实现。

这里是仿真闭环中最重要的“执行器”一层：上层策略给出一个动作后，
环境先交给控制器拆分/转换，再把各关节目标写入 SAPIEN，推进若干底层
物理步，最后重新收集状态、图像、奖励和终止信息。
"""

from typing import Literal

import numpy as np
import sapien
from loguru import logger
from sapien.utils.viewer import Viewer

from galaxea_sim.controllers import *
from galaxea_sim.envs.base.base import SapienEnv
from galaxea_sim.utils.gym_utils import get_observation_space_from_example
from galaxea_sim.robots.bimanual import BimanualRobot

class BimanualManipulationEnv(SapienEnv):
    """支持双臂、双夹爪和多种控制器的 Gym 环境基类。"""

    def __init__(
        self,
        robot_class: type[BimanualRobot],
        robot_kwargs: dict = {},
        controller_type: str = "bimanual_joint_position",
        control_freq: int = 15,
        timestep: float = 0.01,
        headless: bool = True,
        obs_mode: Literal["state", "image"] = "image",  
        ray_tracing: bool = False,
        include_depth: bool = True,
        camera_resolution_scale: int | None = None,
    ):
        """创建机器人、控制器、动作空间和观测空间。

        ``controller_type`` 决定动作的语义：

        * ``bimanual_joint_position``：动作是双臂关节位置和夹爪位置；
        * ``bimanual_ee_pose``：动作是双臂末端位姿和夹爪位置；
        * ``bimanual_relaxed_ik``：动作先经过 Relaxed IK 再变成关节目标。

        OpenPI R1 闭环只允许第一种控制器，因为它要求固定的 14 维动作
        顺序。其它控制器仍可用于原始专家数据和传统评测。
        """
        self.eval_mode = False
        self.robot_name = robot_class.name
        super().__init__(control_freq, timestep, headless, ray_tracing, include_depth)
        robot_kwargs = dict(robot_kwargs)
        if camera_resolution_scale is not None:
            robot_kwargs["camera_resolution_scale"] = camera_resolution_scale
        self.robot: BimanualRobot = robot_class(self._scene, **robot_kwargs)
        self._init_controller(controller_type)
        self._init_buffers()
        self.action_space = self.controller.action_space
        self.obs_mode = obs_mode
        self.observation_space = get_observation_space_from_example(self._get_obs())
        
    def eval(self):
        """切换到评测模式。

        任务子类可以读取 ``eval_mode``，决定是否固定随机性或改变任务
        的可视化行为；基础环境本身不强制使用这个标志。
        """
        self.eval_mode = True
        
    def _build_world(self):
        self._scene.add_ground(0)
        
    @property
    def left_arm_joint_indices(self):
        return self.robot.left_arm_joint_indices
    
    @property
    def right_arm_joint_indices(self):
        return self.robot.right_arm_joint_indices
    
    @property
    def left_gripper_joint_indices(self):
        return self.robot.left_gripper_joint_indices
    
    @property
    def right_gripper_joint_indices(self):
        return self.robot.right_gripper_joint_indices
    
    @property
    def torso_joint_indices(self):
        return self.robot.torso_joint_indices
    
    @property
    def active_joints(self):
        return self.robot.active_joints
    
    @property
    def num_dofs(self):
        return self.robot.num_dofs
    
    @property
    def init_qpos(self):
        return self.robot.init_qpos
    
    @property
    def active_joint_names(self):
        return self.robot.active_joint_names
    
    @property
    def left_ee_link_name(self):
        return self.robot.left_ee_link_name
    
    @property
    def right_ee_link_name(self):
        return self.robot.right_ee_link_name

    def _init_buffers(self):
        """清空上一帧动作缓存。

        这些缓存会被写入下一次观测的 ``upper_body_action_dict``，因此它们
        也是专家数据记录器保存 action 的来源。
        """
        self.last_gripper_cmd = [0, 0]
        self.left_arm_joint_position_cmd = np.zeros(len(self.left_arm_joint_indices))
        self.right_arm_joint_position_cmd = np.zeros(len(self.right_arm_joint_indices))
        self.left_arm_gripper_position_cmd = 0.
        self.right_arm_gripper_position_cmd = 0.
            
    def _init_controller(self, controller_type):
        """根据名称实例化控制器，并保存控制器类型。"""
        self.controller_type = controller_type
        if controller_type == "bimanual_joint_position":
            self.controller = BimanualJointPositionController(self.robot)
        elif controller_type == "bimanual_ee_pose":
            self.controller = BimanualEEPoseController(self.robot)
        elif controller_type == "bimanual_relaxed_ik":
            self.controller = BimanualRelaxedIKController(self.robot)
        else:
            raise ValueError(f"Invalid controller type. Got: {controller_type}")    
        
    def _setup_viewer(self):
        """创建 SAPIEN viewer 并设置默认视角。"""
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.viewer = Viewer(self.renderer)
        self.viewer.plugins[5].show_camera_linesets = False
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=1.2, y=0.25, z=1.5,)
        self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)

    def step(self, action):
        """执行一个控制周期。

        执行顺序必须保持稳定：

        1. 控制器把上层 action 拆成左臂、右臂和两个夹爪的目标；
        2. 没有被控制器管理的关节回到 ``init_qpos`` 目标；
        3. 将目标写入 SAPIEN 的 drive；
        4. 在一个控制周期内推进 ``decimation`` 个物理步；
        5. 采集动作之后的 observation，并计算 reward/termination/info。

        这也是闭环中 ``action_t -> state_{t+1}`` 的边界。
        """
        left_arm_action, left_gripper_action, right_arm_action, right_gripper_action = self.controller.get_control_signal(action)
        self.left_arm_joint_position_cmd = left_arm_action
        self.right_arm_joint_position_cmd = right_arm_action
        self.left_arm_gripper_position_cmd = left_gripper_action
        self.right_arm_gripper_position_cmd = right_gripper_action
        for i in range(self.num_dofs):
            self.active_joints[i].set_drive_target(self.init_qpos[i])
        for i, joint_index in enumerate(self.left_arm_joint_indices):
            self.active_joints[joint_index].set_drive_target(left_arm_action[i])
        for i, joint_index in enumerate(self.right_arm_joint_indices):
            self.active_joints[joint_index].set_drive_target(right_arm_action[i])
        for joint_index, joint_sign in zip(self.left_gripper_joint_indices, self.robot.gripper_finger_sign):
            self.active_joints[joint_index].set_drive_target(left_gripper_action * joint_sign)
        for joint_index, joint_sign in zip(self.right_gripper_joint_indices, self.robot.gripper_finger_sign):
            self.active_joints[joint_index].set_drive_target(right_gripper_action * joint_sign)
        self.last_gripper_cmd = [left_gripper_action, right_gripper_action]
        for i in range(self.decimation):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self._scene.step()

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._check_termination()
        truncated = self._check_truncation()
        info = self._get_info()
        
        self._scene.update_render()

        return obs, reward, terminated, truncated, info
    
    def _get_info(self):
        """返回任务相关的诊断信息；由具体任务覆盖。"""
        return {}
    
    def _get_reset_info(self):
        """返回本次 reset 产生的初始物体状态；由具体任务覆盖。"""
        return {}
    
    def _check_truncation(self):
        """判断是否因时间上限等外部原因截断 episode。"""
        return False
    
    def _check_termination(self) -> bool:
        """判断任务是否已经成功或失败结束；由具体任务覆盖。"""
        return False

    def reset(self, *, seed=None, options=None):
        """重置机器人、控制器和任务世界，并返回初始观测。

        除 Gym 自己的随机数生成器外，这里还同步设置 legacy
        ``numpy.random``，因为现有 RoboTwin 任务的物体随机化代码仍然
        使用全局 NumPy RNG。这样同一个 seed 才能更可靠地复现任务初始状态。
        """
        super().reset(seed=seed)
        # RoboTwin task implementations currently draw from the legacy global
        # NumPy RNG. Seed it as a compatibility bridge so Gym's reset(seed=...)
        # contract is honored until those tasks accept an explicit generator.
        if seed is not None:
            np.random.seed(seed)

        self.robot.set_qpos(self.init_qpos)
        self.robot.set_qvel(np.zeros(self.num_dofs, dtype=np.float32))
        self.robot.set_qf(np.zeros(self.num_dofs, dtype=np.float32))
        for joint, target in zip(self.active_joints, self.init_qpos):
            joint.set_drive_target(target)

        self._init_buffers()
        self.controller.reset()
        reset_info = None if options is None else options.get("reset_info")
        self.reset_world(reset_info=reset_info)
        self._scene.update_render()
        return self._get_obs(), self._get_reset_info()

    def reset_world(self, reset_info=None):
        """由任务类重新摆放任务物体。"""
        pass

    def _get_obs(self):
        """构造统一的双臂观测字典。

        原始观测保留了较丰富的字段，包括关节速度、末端位姿、动作缓存、
        物体状态和语言指令。OpenPI 适配器只从这里挑选三路 RGB、14 维
        状态和 ``language_instruction``，不会把 privileged 的
        ``object_dict`` 发送给策略。
        """
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()
        left_arm_ee_pose = self.robot.left_ee_pose_wrt_control_frame
        right_arm_ee_pose = self.robot.right_ee_pose_wrt_control_frame
        left_arm_joint_position = qpos[self.left_arm_joint_indices]
        right_arm_joint_position = qpos[self.right_arm_joint_indices]
        left_arm_joint_velocity = qvel[self.left_arm_joint_indices]
        right_arm_joint_velocity = qvel[self.right_arm_joint_indices]
        left_arm_gripper_position = qpos[self.left_gripper_joint_indices][0:1]
        right_arm_gripper_position = qpos[self.right_gripper_joint_indices][0:1]
        torso_joint_position = qpos[self.torso_joint_indices]
        
        upper_body_observations = dict(
            left_arm_joint_position=left_arm_joint_position,
            right_arm_joint_position=right_arm_joint_position,
            left_arm_gripper_position=left_arm_gripper_position,
            right_arm_gripper_position=right_arm_gripper_position,          
            left_arm_joint_velocity=left_arm_joint_velocity,
            right_arm_joint_velocity=right_arm_joint_velocity,  
            left_arm_ee_pose=np.concatenate([left_arm_ee_pose.p, left_arm_ee_pose.q]),
            right_arm_ee_pose=np.concatenate([right_arm_ee_pose.p, right_arm_ee_pose.q]),
        )
        upper_body_action_dict = dict(
            left_arm_joint_position_cmd=self.left_arm_joint_position_cmd,
            right_arm_joint_position_cmd=self.right_arm_joint_position_cmd,
            left_arm_gripper_position_cmd=np.array([self.left_arm_gripper_position_cmd]),
            right_arm_gripper_position_cmd=np.array([self.right_arm_gripper_position_cmd]),
        )
        if self.controller_type == "bimanual_ee_pose" or self.controller_type == "bimanual_relaxed_ik":
            upper_body_action_dict.update(
                left_arm_ee_pose_cmd=self.controller.left_arm_cmd,
                right_arm_ee_pose_cmd=self.controller.right_arm_cmd,
            )
        lower_body_observations = dict(
            chassis_joint_position=np.zeros(3),
            torso_joint_position=torso_joint_position,
        )
        lower_body_action_dict = dict(
            chassis_target_speed_cmd=np.zeros(3),
        )
        if self.obs_mode == "image":
            image_dict = self.get_image_dict()
            for key, value in image_dict.items():
                upper_body_observations[key] = value
        object_dict = self.get_object_dict()       
        
        return dict(
            upper_body_observations=upper_body_observations,
            upper_body_action_dict=upper_body_action_dict,
            lower_body_observations=lower_body_observations,
            lower_body_action_dict=lower_body_action_dict,
            language_instruction=self.language_instruction,
            object_dict=object_dict,
        )
        
    def get_object_dict(self):
        """返回任务物体的真值状态；默认没有物体。"""
        return {}
        
    @property
    def language_instruction(self) -> str:
        """返回给语言条件策略的任务指令；由具体任务覆盖。"""
        return ""

    def _get_reward(self):
        """计算当前控制周期奖励；基础实现为零奖励。"""
        return 0.
    
    def solution(self):
        """生成专家规划器使用的高层子任务序列。"""
        substeps = []
        for substep in substeps:
            yield substep
    
    @property
    def cameras(self):
        return self.robot.cameras
