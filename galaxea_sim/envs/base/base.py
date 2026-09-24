"""SAPIEN 仿真环境的最底层基类。

本文件只负责所有任务共有的基础设施：

* 创建 SAPIEN 场景并设置物理时间步长；
* 配置灯光、默认观察相机和可选的 viewer；
* 从挂载在机器人上的相机读取 RGB/Depth 图像；
* 提供 Gym 环境所需的随机种子、渲染和资源查询接口。

具体的机器人、桌子、物体、成功条件和动作控制不在这里实现，而是由
``BimanualManipulationEnv``、R1 机器人类以及各个 RoboTwin 任务类完成。
"""

import sapien.core as sapien
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding

from galaxea_sim.utils.sapien_utils import add_camera_to_scene


class SapienEnv(gym.Env):
    """把一个 SAPIEN 场景包装成 Gymnasium 环境。

    ``timestep`` 是底层物理仿真的时间步长，``control_freq`` 是上层策略
    发送动作的频率。一个策略动作通常不会只推进一次物理步，而是通过
    ``decimation`` 在多个物理步内保持同一个控制目标。
    """

    def __init__(self, control_freq, timestep, headless, ray_tracing, include_depth=True):
        """创建场景、设置渲染参数，并调用子类构建世界。

        Args:
            control_freq: 策略/控制器频率，当前 R1 OpenPI 闭环使用 15 Hz。
            timestep: SAPIEN 的底层物理时间步长，当前默认是 0.01 秒。
            headless: 是否不创建交互式 viewer；服务器批量评测通常为 True。
            ray_tracing: 是否启用 SAPIEN 光线追踪渲染。
            include_depth: 是否在 ``get_image_dict`` 中额外读取深度图。
                π0.5 闭环只使用 RGB，因此闭环路径会关闭它以节省渲染开销。
        """
        if ray_tracing:
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(4)  # change to 256 for less noise
            sapien.render.set_ray_tracing_denoiser("oidn") # change to "optix" or "oidn"

        self.control_freq = control_freq  # alias: frame_skip in mujoco_py
        self.timestep = timestep
        self.headless = headless
        # RGB-only policy inference avoids reading an unused Position buffer.
        # Keep the historical default for data collection and other callers.
        self.include_depth = include_depth

        self._scene = sapien.Scene()
        self._scene.set_timestep(timestep)

        self._build_world()
        self.viewer = None
        self.seed()
        self._add_light()
        self._add_scene_camera()

    @property
    def cameras(self) -> list[sapien.pysapien.render.RenderCameraComponent]:
        """返回当前环境的相机列表。

        基类没有机器人相机，具体机器人（例如 ``R1Robot``）需要覆盖这个
        属性。``get_image_dict`` 会遍历这里返回的相机并生成观测字段。
        """
        return []

    def _add_light(self):
        """配置任务场景通用的环境光、方向光和点光源。"""
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        shadow = True
        direction_lights = [[[0, 1, -1], [0.5, 0.5, 0.5]]]
        for direction_light in direction_lights:
            self._scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
            
        point_lights = [[[1, 2, 2], [1, 1, 1]], [[1, -2, 2], [1, 1, 1]], [[-1, 0, 1], [1, 1, 1]]]
        for point_light in point_lights:
            self._scene.add_point_light(point_light[0], point_light[1], shadow=shadow)

    def _build_world(self):
        """由子类创建地面、桌子、墙和任务物体。"""
        raise NotImplementedError()

    def _setup_viewer(self):
        """由子类创建交互式 viewer。

        无头模式下不会调用该方法；需要 GUI 时，具体环境负责选择 viewer
        的位置、姿态和插件配置。
        """
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Override gym functions
    # ---------------------------------------------------------------------------- #
    def seed(self, seed=None):
        """初始化 Gym 使用的 NumPy 随机数生成器并返回实际 seed。"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """释放环境资源。

        当前项目的 SAPIEN viewer 生命周期由具体环境维护；这里保留 Gym
        的关闭入口，方便上层统一调用 ``env.close()``。
        """
        if self.viewer is not None:
            pass  # release viewer
        
    def _add_scene_camera(self):
        """添加一个与机器人无关的默认观察相机。

        该相机主要用于 ``render`` 的场景预览；策略使用的三路相机来自
        ``self.cameras``，分别是头部相机和左右腕部相机。
        """
        pose = sapien.Pose()
        pose.set_p([1.8, -0.9, 1.])
        pose.set_rpy([0, 0, 2.64])
        self.default_camera = add_camera_to_scene(self._scene, "default_camera", pose=pose)

    def render(self):
        """渲染默认相机并在 GUI 模式下刷新 viewer。

        返回值是 ``H x W x 3`` 的 uint8 RGB 图像。注意：策略闭环不会把
        这个默认相机图像发送给 OpenPI，而是读取机器人挂载的三路相机。
        """
        self.default_camera.take_picture()
        rgba = self.default_camera.get_picture("Color")  # [H, W, 4]
        rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[..., :3]       
        if not self.headless:
            if self.viewer is None:
                self._setup_viewer()
            else:
                if not self.viewer.closed:
                    self.viewer.render()
        return rgb_img

    # ---------------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------------- #
    def get_image_dict(self):
        """采集策略/数据集使用的机器人相机图像。

        RGB 始终以 ``rgb_<camera_name>`` 写入。深度图只有在
        ``include_depth=True`` 时才会读取 Position buffer，并以毫米量纲、
        uint16 类型保存。关闭深度可以避免无用的 Position buffer 读取。
        """
        image_dict = {}
        for camera in self.cameras:
            camera.take_picture()
            rgba = camera.get_picture("Color")  # [H, W, 4]
            rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[..., :3]
            image_dict[f"rgb_{camera.name}"] = rgb_img
            
            if self.include_depth:
                position = camera.get_picture('Position')  # [H, W, 4]
                depth = -position[..., 2]
                depth_image = depth * 1000.0
                depth_image = depth_image.clip(0, 256 * 256 - 1).astype("uint16")
                image_dict[f"depth_{camera.name}"] = depth_image
        return image_dict
    
    def get_actor(self, name):
        """按唯一名字查找场景中的刚体 actor。"""
        all_actors = self._scene.get_all_actors()
        actor = [x for x in all_actors if x.name == name]
        if len(actor) > 1:
            raise RuntimeError(f'Not a unique name for actor: {name}')
        elif len(actor) == 0:
            raise RuntimeError(f'Actor not found: {name}')
        return actor[0]

    def get_articulation(self, name):
        """按唯一名字查找场景中的 articulation。"""
        all_articulations = self._scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f'Not a unique name for articulation: {name}')
        elif len(articulation) == 0:
            raise RuntimeError(f'Articulation not found: {name}')
        return articulation[0]

    @property
    def dt(self):
        """一个控制周期对应的仿真时间，单位为秒。"""
        return 1 / self.control_freq
    
    @property
    def decimation(self):
        """一个策略动作需要推进的底层物理步数。"""
        return int(self.dt / self.timestep)
