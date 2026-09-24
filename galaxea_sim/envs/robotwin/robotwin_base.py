"""RoboTwin 任务共享的场景基类。

这里把多个具体任务都会用到的公共场景元素集中起来：

* 创建双臂机器人所在的桌面；
* 创建桌面后方的墙体，给相机提供稳定的视觉背景；
* 根据机器人型号选择不同的桌面高度和桌面中心位置；
* 配置仿真的环境光、方向光和点光源。

具体任务（例如双瓶抓取、方块堆叠）只需要继承本类，再在自己的文件中
增加物体、初始状态、成功判定和任务专属 reset 逻辑即可。这个分层也解释了
为什么 `gym.make("某个任务 ID")` 最终会同时得到“机器人 + 通用桌面场景 +
任务物体”的完整环境。
"""

import sapien
import numpy as np

from typing import Literal

from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.utils.robotwin_utils import create_table, create_box

class RoboTwinBaseEnv(BimanualManipulationEnv):
    """RoboTwin 双臂操作任务的公共场景基类。

    本类本身通常不是用户直接运行的任务，而是具体任务环境的父类。任务注册
    文件会把 `robot_class`、初始关节角和任务入口传进来；父类负责通用仿真
    生命周期，子类负责任务物体和任务成功条件。

    需要特别区分两个概念：

    * `variant_idx` 是同一个任务的场景变体编号，影响任务随机化或物体布局；
    * Gym 的环境 ID（例如 `R1DualBottlesPickEasy-v0`）才是外部选择任务时
      使用的名字，通常在同目录的 `__init__.py` 中注册。
    """

    def __init__(
        self,
        robot_class,
        robot_kwargs: dict = {},
        controller_type: str = "bimanual_joint_position",
        control_freq: int = 15,
        timestep: float = 0.01,
        headless: bool = True,
        obs_mode: Literal["state", "image"] = "image",  
        variant_idx: int = 0,
        ray_tracing: bool = False,
        include_depth: bool = True,
        camera_resolution_scale: int | None = None,
    ):
        """保存场景变体并初始化通用双臂操作环境。

        `super().__init__` 会继续创建 SAPIEN 场景、机器人、控制器以及相机。
        当前类不在这里直接创建桌面，因为桌面依赖于机器人型号；统一的场景
        构建入口是 `_build_world`，它会在父类基础设施完成后补上桌面和墙体。
        """
        self.variant_idx = variant_idx
        super().__init__(
            robot_class,
            robot_kwargs,
            controller_type,
            control_freq,
            timestep,
            headless,
            obs_mode,
            ray_tracing,
            include_depth,
            camera_resolution_scale,
        )
    
    def _build_world(self):
        """创建通用世界：父类基础设施、桌面、墙体和初始世界状态。"""
        super()._build_world()
        self._setup_table()
        self._setup_wall()
        self.reset_world()

    @property
    def table_length(self):
        """桌面沿机器人前后方向的长度，单位为米。"""
        return 1.2
    
    @property
    def table_width(self):
        """桌面沿机器人左右方向的宽度，单位为米。"""
        return 0.7
    
    @property
    def table_height(self):
        """桌面高度，R1 Lite 与其他机器人型号使用不同高度。"""
        if self.robot_name == "r1_lite":
            return 0.7
        else:
            return 0.9
        
    @property
    def tabletop_center_x(self):
        """桌面中心在世界坐标系 X 轴上的位置，单位为米。"""
        if self.robot_name == "r1_lite":
            return 0.5
        else:
            return 0.7

    def _setup_table(self):
        """在 SAPIEN 场景中创建静态桌面。

        `tabletop_center_in_world` 是后续放置任务物体时常用的参考坐标；任务
        文件一般会在此基础上叠加偏移，而不会重新猜测桌面位置。
        """
        self.table_static = True
        self.tabletop_center_in_world = np.array([self.tabletop_center_x, 0, self.table_height])
        self.table = create_table(
            self._scene,
            sapien.Pose(p=self.tabletop_center_in_world),
            length=self.table_length,
            width=self.table_width,
            height=self.table_height,
            thickness=0.05,
            is_static=self.table_static
        )

    def _setup_wall(self):
        """在桌面后方创建静态背景墙，避免场景背景为空。"""
        self.wall = create_box(
            self._scene,
            sapien.Pose(p=[1.0 + self.tabletop_center_in_world[0], 0, 1.5]),
            half_size=[0.6, 3, 1.5],
            color=(1, 0.9, 0.9), 
            name='wall',
        )
        
    def _add_light(self):
        """配置环境光、方向光和左右两盏点光源。

        光照只影响渲染观测，不参与物理控制；但它会直接影响相机图像，因此
        对视觉策略的输入分布很重要。任务变体通常不应随意改变这里的光照，
        否则训练数据与推理时的图像分布可能不一致。
        """
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        shadow = True
        direction_lights = [[[0.5 + 0.7, 0, -1], [0.5, 0.5, 0.5]]]
        for direction_light in direction_lights:
            self._scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        point_lights = [[[0.7, 1, 1.8], [1, 1, 1]], [[0.7, -1, 1.8], [1, 1, 1]]]
        for point_light in point_lights:
            self._scene.add_point_light(point_light[0], point_light[1], shadow=shadow)
