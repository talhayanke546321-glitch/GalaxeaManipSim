"""使用内置双臂专家规划器采集仿真示范轨迹。

采集器不是 OpenPI 在线闭环，而是离线数据生产环节：任务类生成高层
substep，``BimanualPlanner`` 将其解析为一串关节位置动作。每一步先
记录 ``state_t + action_t``，再调用 ``env.step(action_t)``，保证后续
行为克隆训练的时间对齐。
"""

import json
import re

from typing import Literal, Optional

import gymnasium as gym
import tyro
import cv2
import numpy as np
from loguru import logger
from pathlib import Path

import galaxea_sim.envs  # noqa: F401 - registers the Gym environments
from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.planners.bimanual import BimanualPlanner
from galaxea_sim.utils.data_utils import (
    PRE_ACTION_RECORDING_CONTRACT,
    record_observation_action,
    save_dict_list_to_hdf5,
    save_dict_list_to_json,
)
from galaxea_sim.utils.image_utils import prepare_rgb_image


def _compact_observation_images(observation: dict) -> dict:
    """把离线记录的 RGB 图像处理成和在线推理完全相同的 224x224 契约。"""
    upper = observation["upper_body_observations"]
    for key in ("rgb_head", "rgb_left_hand", "rgb_right_hand"):
        upper[key] = prepare_rgb_image(upper[key], 224, 224, name=key)
    # Depth is not sent to π0.5. Keep this tolerant of the OpenPI collection
    # path, which disables depth rendering entirely.
    for key in ("depth_head", "depth_left_hand", "depth_right_hand"):
        if key in upper:
            depth = np.asarray(upper[key])
            if depth.shape != (224, 224):
                upper[key] = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)
    return observation

def main(
    env_name: str, 
    num_demos: int = 100, 
    dataset_dir: str = 'datasets', 
    control_freq: int = 15, 
    headless: bool = True, 
    obs_mode: Literal['state', 'image'] = 'state', 
    tag: Optional[str] = 'collected', 
    ray_tracing: bool = False,
    seed: Optional[int] = None,
    max_tries: Optional[int] = None,
    store_resized_images: bool = False,
    table_height: Optional[float] = None,
):
    """按任务、数量和 seed 采集成功的专家 episode。"""
    if num_demos < 0:
        raise ValueError("num_demos must be non-negative")
    if control_freq <= 0:
        raise ValueError("control_freq must be positive")
    if store_resized_images and obs_mode != "image":
        raise ValueError("--store-resized-images requires --obs-mode=image")
    if max_tries is not None and max_tries < 0:
        raise ValueError("max_tries must be non-negative")
    if table_height is not None and not 0.70 <= table_height <= 1.10:
        raise ValueError("table_height must be between 0.70 and 1.10 meters")

    # 采集 OpenPI 数据时关闭深度，并让相机分辨率保持与在线推理一致；
    # 非 OpenPI 的旧数据流程仍可通过默认参数保留原始深度行为。
    env = gym.make(
        env_name,
        control_freq=control_freq,
        headless=headless,
        obs_mode=obs_mode,
        ray_tracing=ray_tracing,
        include_depth=False,
        table_height_override=table_height,
    )
    assert isinstance(env.unwrapped, BimanualManipulationEnv)
    planner = BimanualPlanner(
        urdf_path=f"{env.unwrapped.robot.name}/robot.urdf",
        srdf_path=None,
        left_arm_move_group=env.unwrapped.left_ee_link_name,
        right_arm_move_group=env.unwrapped.right_ee_link_name,
        active_joint_names=env.unwrapped.active_joint_names,
        control_freq=env.unwrapped.control_freq,
    )

    save_dir = Path(dataset_dir) / env_name / (tag or "collected")
    existing_paths = sorted(save_dir.glob("demo_*.h5"))
    index_pattern = re.compile(r"demo_(\d+)\.h5$")
    existing_indices = [
        int(match.group(1))
        for path in existing_paths
        if (match := index_pattern.match(path.name)) is not None
    ]
    num_collected = len(existing_indices)
    next_index = max(existing_indices, default=-1) + 1
    meta_path = save_dir / "meta_info.json"
    if meta_path.exists():
        with meta_path.open() as file:
            meta_info_list = json.load(file)
    else:
        meta_info_list = []
    previous_tries = (
        int(meta_info_list[-1].get("num_tries", num_collected))
        if meta_info_list
        else num_collected
    )
    num_tries = max(previous_tries, num_collected)
    max_tries = max_tries if max_tries is not None else max(num_demos * 20, num_demos)
    while num_collected < num_demos and num_tries < max_tries:
        num_steps = 0
        traj = []
        info = {}
        episode_seed = None if seed is None else seed + num_tries
        obs, rest_info = env.reset(seed=episode_seed)
        if not headless:
            env.render()
        episode_done = False
        for substep in env.unwrapped.solution():
            actions = planner.solve(
                substep, env.unwrapped.robot.get_qpos(), env.unwrapped.last_gripper_cmd, 
                verbose=False
            )
            if actions is not None:
                for action in actions:
                    # 必须在推进仿真之前记录 state_t/action_t。若先 step 再
                    # 记录，会把动作错配到 state_{t+1}，训练出的策略会出现
                    # 一步延迟。
                    recorded_obs = record_observation_action(
                        obs,
                        action,
                        controller_type="bimanual_joint_position",
                    )
                    if store_resized_images:
                        recorded_obs = _compact_observation_images(recorded_obs)
                    traj.append(recorded_obs)
                    num_steps += 1
                    obs, _, terminated, truncated, info = env.step(action)
                    if not headless:
                        env.render()
                    if terminated or truncated:
                        episode_done = True
                        break
            if episode_done:
                break
        num_tries += 1
        if info.get("success", False):
            save_dict_list_to_hdf5(
                traj,
                save_dir / f"demo_{next_index}.h5",
                metadata={
                    "recording_contract": PRE_ACTION_RECORDING_CONTRACT,
                    "controller_type": "bimanual_joint_position",
                    "control_freq": int(control_freq),
                    "camera_resolution_scale": int(
                        getattr(env.unwrapped.robot, "camera_resolution_scale", 1)
                    ),
                    "table_height_m": float(env.unwrapped.table_height),
                },
            )
            next_index += 1
            num_collected += 1
            meta_info = dict(reset_info=rest_info, success=info["success"], total_steps=num_steps, num_collected=num_collected, num_tries=num_tries)
            meta_info_list.append(meta_info)
            save_dict_list_to_json(meta_info_list, save_dir / "meta_info.json")
            success_rate = int(num_collected / num_tries * 100) if num_tries else 0
            logger.info(
                f"Collected {num_collected} demos in {num_tries} tries. "
                f"Success rate: {success_rate}%"
            )

    env.close()
    if num_collected < num_demos:
        raise RuntimeError(
            f"Collected {num_collected}/{num_demos} demos after {num_tries} tries for {env_name}"
        )

if __name__ == "__main__":
    tyro.cli(main)
