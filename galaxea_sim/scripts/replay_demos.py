import gymnasium as gym
import h5py
import json
import numpy as np
import re
import time
import tyro
import tqdm
from loguru import logger
from pathlib import Path

import galaxea_sim.envs  # noqa: F401 - registers the Gym environments
from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.utils.data_utils import (
    PRE_ACTION_RECORDING_CONTRACT,
    record_observation_action,
    save_dict_list_to_hdf5,
    save_dict_list_to_json,
)
from galaxea_sim.utils.image_utils import prepare_rgb_image


_DEMO_NAME_PATTERN = re.compile(r"demo_(\d+)\.h5$")


def _compact_observation_images(observation: dict) -> dict:
    """Keep replayed episodes on the same 224x224 RGB contract as collection."""
    upper = observation["upper_body_observations"]
    for key in ("rgb_head", "rgb_left_hand", "rgb_right_hand"):
        upper[key] = prepare_rgb_image(upper[key], 224, 224, name=key)
    return observation


def _demo_index(path: Path) -> int | None:
    match = _DEMO_NAME_PATTERN.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def _read_source_reset_info(h5_path: Path, demo_index: int) -> dict:
    meta_info_path = h5_path.parent / "meta_info.json"
    if not meta_info_path.exists():
        return {}
    with meta_info_path.open() as file:
        source_meta_info_list = json.load(file)
    if not isinstance(source_meta_info_list, list) or demo_index >= len(source_meta_info_list):
        raise ValueError(
            f"Source metadata does not contain demo_{demo_index} for {h5_path}"
        )
    reset_info = source_meta_info_list[demo_index].get("reset_info", {})
    return reset_info if isinstance(reset_info, dict) else {}


def _read_actions(h5_file: h5py.File, target_controller_type: str) -> np.ndarray:
    upper = h5_file["upper_body_observations"]
    commands = h5_file["upper_body_action_dict"]
    left_ee_pose = upper["left_arm_ee_pose"][()]
    right_ee_pose = upper["right_arm_ee_pose"][()]
    left_joint_cmd = commands["left_arm_joint_position_cmd"][()]
    right_joint_cmd = commands["right_arm_joint_position_cmd"][()]
    left_gripper_cmd = commands["left_arm_gripper_position_cmd"][()]
    right_gripper_cmd = commands["right_arm_gripper_position_cmd"][()]

    arrays = {
        "right_arm_ee_pose": right_ee_pose,
        "left_arm_joint_position_cmd": left_joint_cmd,
        "right_arm_joint_position_cmd": right_joint_cmd,
        "left_arm_gripper_position_cmd": left_gripper_cmd,
        "right_arm_gripper_position_cmd": right_gripper_cmd,
    }
    episode_length = len(left_ee_pose)
    if episode_length == 0:
        raise ValueError("Cannot replay an empty episode")
    for name, array in arrays.items():
        if len(array) != episode_length:
            raise ValueError(
                f"Episode arrays have inconsistent lengths: {name} has {len(array)}, "
                f"expected {episode_length}"
            )

    if target_controller_type == "bimanual_joint_position":
        return np.concatenate(
            [left_joint_cmd, left_gripper_cmd, right_joint_cmd, right_gripper_cmd],
            axis=-1,
        )
    if target_controller_type in {"bimanual_ee_pose", "bimanual_relaxed_ik"}:
        return np.concatenate(
            [left_ee_pose, left_gripper_cmd, right_ee_pose, right_gripper_cmd],
            axis=-1,
        )
    raise ValueError(f"Unknown target controller type: {target_controller_type}")


def main(
    env_name: str,
    num_demos: int = 100,
    dataset_dir: str = "datasets",
    target_controller_type: str = "bimanual_relaxed_ik",
    control_freq: int = 15,
    headless: bool = True,
    realtime: bool = False,
    ray_tracing: bool = False,
):
    if num_demos < 0:
        raise ValueError("num_demos must be non-negative")
    if control_freq <= 0:
        raise ValueError("control_freq must be positive")
    if target_controller_type not in {
        "bimanual_joint_position",
        "bimanual_ee_pose",
        "bimanual_relaxed_ik",
    }:
        raise ValueError(f"Unknown target controller type: {target_controller_type}")

    env = gym.make(
        env_name,
        control_freq=control_freq,
        headless=headless,
        controller_type=target_controller_type,
        ray_tracing=ray_tracing,
        include_depth=False,
    )
    assert isinstance(env.unwrapped, BimanualManipulationEnv)
    save_dir = Path(dataset_dir) / env_name / "replayed"
    source_dir = Path(dataset_dir) / env_name
    h5_paths = sorted(
        h5_path
        for h5_path in source_dir.glob("*/*.h5")
        if h5_path.parent.name not in {"final", "replayed"}
    )
    num_collected = 0
    num_tries = 0
    logger.info(f"Collecting {num_demos} demos from {len(h5_paths)} h5 files.")
    existing = sorted(save_dir.glob("demo_*.h5"))
    existing_indices = [index for path in existing if (index := _demo_index(path)) is not None]
    num_collected = len(existing_indices)
    next_index = max(existing_indices, default=-1) + 1

    meta_info_path = save_dir / "meta_info.json"
    if meta_info_path.exists():
        with meta_info_path.open() as file:
            meta_info_list = json.load(file)
        if not isinstance(meta_info_list, list):
            raise ValueError(f"Replay metadata must be a list: {meta_info_path}")
    else:
        meta_info_list = []

    # Output numbering and source numbering are different namespaces.  They
    # only happen to match when every source episode replays successfully. If
    # one episode fails, filtering by output indices would replay successful
    # source episodes again on resume.  Use the recorded source_demo mapping;
    # retain the old index-based fallback only for pre-contract output dirs
    # that have no replay metadata at all.
    processed_source_indices = {
        int(item["source_demo"])
        for item in meta_info_list
        if isinstance(item, dict) and item.get("source_demo") is not None
    }
    if not meta_info_list and existing_indices:
        processed_source_indices = set(existing_indices)
    h5_paths = [
        path
        for path in h5_paths
        if (index := _demo_index(path)) is not None and index not in processed_source_indices
    ]

    pbar = tqdm.tqdm(
        total=max(num_demos, num_collected),
        initial=num_collected,
        desc="Collecting demos",
    )
    try:
        for h5_path in h5_paths:
            if num_collected >= num_demos:
                break
            demo_idx = _demo_index(h5_path)
            assert demo_idx is not None
            reset_info = _read_source_reset_info(h5_path, demo_idx)

            with h5py.File(h5_path, "r") as h5_file:
                actions = _read_actions(h5_file, target_controller_type)
                if reset_info:
                    obs, _ = env.reset(options={"reset_info": reset_info})
                else:
                    obs, _ = env.reset()
                if not headless:
                    env.render()
                    if realtime:
                        time.sleep(1 / control_freq)

                traj = []
                info = {}
                episode_done = False
                for action in actions:
                    recorded_obs = record_observation_action(
                        obs,
                        action,
                        controller_type=target_controller_type,
                    )
                    traj.append(_compact_observation_images(recorded_obs))
                    obs, _, terminated, truncated, info = env.step(action)
                    if not headless:
                        env.render()
                        if realtime:
                            time.sleep(1 / control_freq)
                    episode_done = bool(terminated or truncated)
                    if episode_done:
                        break
                if not episode_done:
                    for _ in range(5):
                        recorded_obs = record_observation_action(
                            obs,
                            actions[-1],
                            controller_type=target_controller_type,
                        )
                        traj.append(_compact_observation_images(recorded_obs))
                        obs, _, terminated, truncated, info = env.step(actions[-1])
                        if not headless:
                            env.render()
                            if realtime:
                                time.sleep(1 / control_freq)
                        if terminated or truncated:
                            break

            num_tries += 1
            if info.get("success", False):
                output_index = next_index
                save_dict_list_to_hdf5(
                    traj,
                    save_dir / f"demo_{output_index}.h5",
                    metadata={
                        "recording_contract": PRE_ACTION_RECORDING_CONTRACT,
                        "controller_type": target_controller_type,
                        "control_freq": int(control_freq),
                        "camera_resolution_scale": int(
                            getattr(env.unwrapped.robot, "camera_resolution_scale", 1)
                        ),
                    },
                )
                next_index += 1
                num_collected += 1
                meta_info_list.append(
                    dict(
                        source_demo=demo_idx,
                        reset_info=reset_info,
                        success=info["success"],
                        total_steps=len(traj),
                        num_collected=num_collected,
                    )
                )
                save_dict_list_to_json(meta_info_list, meta_info_path)
                pbar.update(1)
                success_rate = int(num_collected / num_tries * 100) if num_tries else 0
                pbar.set_postfix_str(
                    f"Collected {num_collected} demos in {num_tries} tries. "
                    f"Success rate: {success_rate}%"
                )

    finally:
        pbar.close()
        env.close()

    if num_collected < num_demos:
        success_rate = int(num_collected / num_tries * 100) if num_tries else 0
        logger.warning(
            f"Collected {num_collected} demos in {num_tries} tries. "
            f"Success rate: {success_rate}%"
        )
        logger.warning(f"Failed to collect {num_demos - num_collected} demos.")

if __name__ == "__main__":
    tyro.cli(main)
