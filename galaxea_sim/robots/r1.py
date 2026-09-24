"""标准 Galaxea R1 机器人模型和三路 RGB 相机配置。"""

import numpy as np
import sapien

from galaxea_sim.utils.sapien_utils import get_link_by_name, add_mounted_camera_to_scene

from .bimanual import BimanualRobot

class R1Robot(BimanualRobot):
    """标准 R1 的双臂 URDF、初始姿态和头部/腕部相机。"""

    name: str = "r1"
    def __init__(
        self, 
        scene: sapien.Scene,
        urdf_path = "r1/robot.urdf",
        robot_origin_xyz = [0, 0, 0],
        robot_origin_quat = [1, 0, 0, 0],
        joint_stiffness = 1000,
        joint_damping = 200,       
        init_qpos = [
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0,
        ],
        touch_link_names = [
            "left_gripper_finger_link1", "left_gripper_finger_link2",
            "right_gripper_finger_link1", "right_gripper_finger_link2"
        ],
        left_arm_joint_key: str = "left_arm",
        right_arm_joint_key: str = "right_arm",
        torso_joint_key: str = "torso",
        left_gripper_joint_key: str = "left_gripper",
        right_gripper_joint_key: str = "right_gripper",
        left_relaxed_ik_setting_path: str = "r1/configs/settings_left.yaml",
        right_relaxed_ik_setting_path: str = "r1/configs/settings_right.yaml",
        camera_resolution_scale: int = 4,
    ):
        """初始化 R1 并保存相机分辨率缩放系数。

        头部相机的基础分辨率是 320x180，``camera_resolution_scale=4``
        时渲染为 1280x720；左右腕部相机仍保持其原始 320x240 配置。
        之后由 OpenPI 图像适配器统一缩放/补黑边到 224x224。
        """
        if camera_resolution_scale <= 0:
            raise ValueError("camera_resolution_scale must be positive")
        self.camera_resolution_scale = camera_resolution_scale
        super().__init__(
            scene,
            urdf_path,
            robot_origin_xyz,
            robot_origin_quat,
            joint_stiffness,
            joint_damping,
            init_qpos,
            touch_link_names,
            left_arm_joint_key,
            right_arm_joint_key,
            torso_joint_key,
            left_gripper_joint_key,
            right_gripper_joint_key,
            left_relaxed_ik_setting_path=left_relaxed_ik_setting_path,
            right_relaxed_ik_setting_path=right_relaxed_ik_setting_path,
        )
        
    def _add_sensors(self):
        """把头部、左腕和右腕相机挂载到对应的 URDF link 上。"""
        self.head_camera = add_mounted_camera_to_scene(
            scene=self._scene,
            mount=get_link_by_name(self.links, "zed_link").entity,
            name="head",
            width=320 * self.camera_resolution_scale,
            height=180 * self.camera_resolution_scale,
            local_pose=sapien.Pose([0, 0, 0], np.array([1, 1, -1, 1]) / 2),
        )
        self.head_camera.set_fovx(np.deg2rad(100.83704311108994))
        self.head_camera.set_fovy(np.deg2rad(68.99805528259))
        alpha = -10
        alpha = np.deg2rad(alpha)
        R = sapien.Pose([0, 0, 0])
        R.set_rpy([alpha, 0, 0])
        self.left_wrist_camera = add_mounted_camera_to_scene(
            scene=self._scene,
            mount=get_link_by_name(self.links, "left_realsense_link").entity,
            name="left_hand",
            width=320,
            height=240,
            local_pose=R*sapien.Pose([0, 0, -0.0], [0.5, 0.5, -0.5, 0.5]),
        )
        self.left_wrist_camera.set_fovx(np.deg2rad(54.39320914794245))
        self.left_wrist_camera.set_fovy(np.deg2rad(43.973014784873506))
        
        self.right_wrist_camera = add_mounted_camera_to_scene(
            scene=self._scene,
            mount=get_link_by_name(self.links, "right_realsense_link").entity,
            name="right_hand",
            width=320,
            height=240,
            local_pose=R*sapien.Pose([0, 0, -0.0], [0.5, 0.5, -0.5, 0.5]),
        )
        self.right_wrist_camera.set_fovx(np.deg2rad(55.702802294064554))
        self.right_wrist_camera.set_fovy(np.deg2rad(44.5846756133851))
        
    @property
    def cameras(self):
        """按 OpenPI 输入顺序返回三路相机：头部、左腕、右腕。"""
        return [self.head_camera, self.left_wrist_camera, self.right_wrist_camera]
