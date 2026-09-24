"""把符合契约的 Galaxea HDF5 轨迹转换成 OpenPI LeRobot 数据集。

这个转换器只接受标准 R1 + ``bimanual_joint_position`` + 15 Hz +
``pre_action_v1`` 数据。它把仿真器中的嵌套字段拆成 LeRobot 的独立
feature，再由 OpenPI 的 ``GalaxeaLeRobotRepack`` 在训练时重新拼回
14 维接口。这样离线训练和在线推理可以共享同一套状态/动作语义。
"""

from __future__ import annotations

import glob
from pathlib import Path
import re
import shutil
from typing import Literal

import h5py
import numpy as np
import tyro

try:
    # OpenPI's pinned LeRobot layout. Prefer it so a dataset generated for
    # OpenPI is not accidentally written with the simulator environment's
    # incompatible LeRobot/datasets versions.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _LEROBOT_TASK_ARGUMENT = False
except ModuleNotFoundError:
    # Galaxea's simulator-pinned LeRobot fork remains a fallback for users who
    # run this converter from the standalone simulator environment.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _LEROBOT_TASK_ARGUMENT = True

from galaxea_sim.utils.data_utils import (
    IMAGE_PREPROCESS_CONTRACT,
    PRE_ACTION_RECORDING_CONTRACT,
    validate_recording_contract,
)
from galaxea_sim.utils.image_utils import prepare_rgb_image

IMAGE_SHAPE = (224, 224, 3)
ARM_DOF = 6
ACTION_DIM = 14
CONTROL_FREQ = 15
_TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_task_name(task: str) -> None:
    """阻止任务名穿越输出目录路径。"""
    if not _TASK_NAME_PATTERN.fullmatch(task):
        raise ValueError(
            "task names must be a single path-safe component containing only "
            "letters, numbers, '.', '_' or '-': "
            f"{task!r}"
        )


def _decode_instruction(value: object) -> str:
    """把 HDF5 中的 bytes/字符串统一解码成 Python str。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _episode_instruction(h5_file: h5py.File, fallback: str) -> str:
    """读取一条 episode 的固定语言指令，并检查整条轨迹保持不变。"""
    if "language_instruction" not in h5_file:
        return fallback
    values = h5_file["language_instruction"][()]
    if len(values) == 0:
        return fallback
    instruction = _decode_instruction(values[0])
    if any(_decode_instruction(value) != instruction for value in values[1:]):
        raise ValueError("language_instruction must be constant within one episode")
    return instruction


def _h5_paths(dataset_dir: str, task: str, tag: str | None) -> list[str]:
    """按任务和 tag 收集待转换的 HDF5 episode。"""
    if tag:
        pattern = f"{dataset_dir}/{task}/{tag}/*.h5"
    else:
        pattern = f"{dataset_dir}/{task}/**/*.h5"
    return sorted(glob.glob(pattern, recursive=True))


def _features(use_video: bool) -> dict[str, dict[str, object]]:
    """声明 LeRobot 数据集的图片、状态和动作 feature 形状。"""
    image_dtype = "video" if use_video else "image"
    return {
        "observation.images.head_rgb": {
            "dtype": image_dtype,
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.left_wrist_rgb": {
            "dtype": image_dtype,
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_wrist_rgb": {
            "dtype": image_dtype,
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.state.left_arm_joints": {
            "dtype": "float32",
            "shape": (ARM_DOF,),
            "names": None,
        },
        "observation.state.left_gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "observation.state.right_arm_joints": {
            "dtype": "float32",
            "shape": (ARM_DOF,),
            "names": None,
        },
        "observation.state.right_gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "action.left_arm_joints": {
            "dtype": "float32",
            "shape": (ARM_DOF,),
            "names": None,
        },
        "action.left_gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "action.right_arm_joints": {
            "dtype": "float32",
            "shape": (ARM_DOF,),
            "names": None,
        },
        "action.right_gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }


def _add_frame(dataset: LeRobotDataset, frame: dict, instruction: str) -> None:
    """兼容不同 LeRobot API，把语言指令写入 task 字段。"""
    if _LEROBOT_TASK_ARGUMENT:
        dataset.add_frame(frame, task=instruction)
    else:
        dataset.add_frame({**frame, "task": instruction})


def _validate_episode_lengths(
    h5_path: str,
    episode_length: int,
    arrays: dict[str, np.ndarray],
) -> None:
    """确保一条 episode 的所有字段拥有相同帧数。"""
    for name, array in arrays.items():
        if len(array) != episode_length:
            raise ValueError(
                f"Episode arrays have inconsistent lengths in {h5_path}: "
                f"{name} has {len(array)}, expected {episode_length}"
            )


def _validate_r1_values(h5_path: str, state: np.ndarray, action: np.ndarray) -> None:
    """校验 R1 状态/动作维度、有限性以及夹爪米制范围。"""
    if state.shape[1:] != (ACTION_DIM,) or action.shape[1:] != (ACTION_DIM,):
        raise ValueError(
            f"R1 state/action must have trailing shape ({ACTION_DIM},) in {h5_path}; "
            f"got {state.shape} and {action.shape}"
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError(f"R1 state/action contains NaN or Inf in {h5_path}")
    for values, name in ((state[:, (6, 13)], "state"), (action[:, (6, 13)], "action")):
        if np.any(values < -1e-6) or np.any(values > 0.05 + 1e-6):
            raise ValueError(f"R1 {name} grippers are outside [0, 0.05] metres in {h5_path}")


def main(
    task: str,
    dataset_dir: str = "datasets",
    tag: str | None = None,
    robot: Literal["r1", "r1_pro", "r1_lite"] = "r1",
    use_eef: bool = False,
    use_video: bool = False,
    push_to_hub: bool = False,
    source_tasks: list[str] | None = None,
    overwrite: bool = False,
):
    """从一个或多个标准 R1 任务目录创建 LeRobot 数据集。

    Create ``task`` from one or more standard-R1 joint-position task dirs.

    ``source_tasks`` enables language-conditioned multitask conversion. Each
    episode's constant ``language_instruction`` becomes its LeRobot task.
    The OpenPI adapter intentionally supports only standard R1 joint-position
    data; the original converter remains available for other combinations.
    """

    # OpenPI 当前的动作契约是标准 R1 的 14 维关节位置；R1 Pro、R1 Lite
    # 或 EEF 数据仍可走原始 LeRobot 转换器，但不能混入这个数据集。
    if robot != "r1" or use_eef:
        raise ValueError(
            "The OpenPI Galaxea adapter supports standard R1 joint-position data only; "
            "use the original converter for --robot=r1_pro/r1_lite or --use-eef"
        )
    _validate_task_name(task)
    source_tasks = source_tasks or [task]
    for source_task in source_tasks:
        _validate_task_name(source_task)

    episodes: list[tuple[str, str]] = []
    for source_task in source_tasks:
        h5_paths = _h5_paths(dataset_dir, source_task, tag)
        if not h5_paths:
            raise FileNotFoundError(f"No HDF5 episodes found for source task {source_task!r}")
        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as h5_file:
                validate_recording_contract(
                    h5_file.attrs,
                    expected_controller_types={"bimanual_joint_position"},
                    expected_control_freq=CONTROL_FREQ,
                    expected_camera_resolution_scale=4,
                )
                episodes.append((h5_path, _episode_instruction(h5_file, fallback=source_task)))

    output_root = Path(dataset_dir).expanduser() / task / "lerobot"
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists at {output_root}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    # 输出的 feature 以组件字段保存，训练时由 GalaxeaLeRobotRepack 重新
    # 拼接；这样 LeRobot metadata 清楚记录每个关节/夹爪字段的含义。
    dataset = LeRobotDataset.create(
        repo_id=task,
        fps=CONTROL_FREQ,
        root=output_root,
        robot_type="r1",
        features=_features(use_video),
        use_videos=use_video,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    dataset.meta.info.update(
        {
            "galaxea_recording_contract": PRE_ACTION_RECORDING_CONTRACT,
            "galaxea_controller_type": "bimanual_joint_position",
            "galaxea_image_preprocess": IMAGE_PREPROCESS_CONTRACT,
            "galaxea_robot": "r1",
            "galaxea_action_dim": ACTION_DIM,
            "galaxea_control_freq": CONTROL_FREQ,
            "galaxea_camera_resolution_scale": 4,
        }
    )

    for h5_path, episode_instruction in episodes:
        with h5py.File(h5_path, "r") as h5_file:
            upper = h5_file["upper_body_observations"]
            commands = h5_file["upper_body_action_dict"]
            rgb_head = upper["rgb_head"][()]
            rgb_left_hand = upper["rgb_left_hand"][()]
            rgb_right_hand = upper["rgb_right_hand"][()]
            left_arm_joints = upper["left_arm_joint_position"][()]
            left_gripper = upper["left_arm_gripper_position"][()]
            right_arm_joints = upper["right_arm_joint_position"][()]
            right_gripper = upper["right_arm_gripper_position"][()]
            left_arm_action = commands["left_arm_joint_position_cmd"][()]
            left_gripper_action = commands["left_arm_gripper_position_cmd"][()]
            right_arm_action = commands["right_arm_joint_position_cmd"][()]
            right_gripper_action = commands["right_arm_gripper_position_cmd"][()]

            episode_length = len(rgb_head)
            if episode_length == 0:
                raise ValueError(f"Episode is empty: {h5_path}")
            _validate_episode_lengths(
                h5_path,
                episode_length,
                {
                    "rgb_left_hand": rgb_left_hand,
                    "rgb_right_hand": rgb_right_hand,
                    "left_arm_joints": left_arm_joints,
                    "left_gripper": left_gripper,
                    "right_arm_joints": right_arm_joints,
                    "right_gripper": right_gripper,
                    "left_arm_action": left_arm_action,
                    "left_gripper_action": left_gripper_action,
                    "right_arm_action": right_arm_action,
                    "right_gripper_action": right_gripper_action,
                },
            )
            state = np.concatenate(
                [left_arm_joints, left_gripper, right_arm_joints, right_gripper], axis=-1
            ).astype(np.float32, copy=False)
            action = np.concatenate(
                [left_arm_action, left_gripper_action, right_arm_action, right_gripper_action], axis=-1
            ).astype(np.float32, copy=False)
            _validate_r1_values(h5_path, state, action)

            rgb_head_resized = [
                prepare_rgb_image(image, IMAGE_SHAPE[0], IMAGE_SHAPE[1], name="rgb_head")
                for image in rgb_head
            ]
            rgb_left_hand_resized = [
                prepare_rgb_image(image, IMAGE_SHAPE[0], IMAGE_SHAPE[1], name="rgb_left_hand")
                for image in rgb_left_hand
            ]
            rgb_right_hand_resized = [
                prepare_rgb_image(image, IMAGE_SHAPE[0], IMAGE_SHAPE[1], name="rgb_right_hand")
                for image in rgb_right_hand
            ]

            for index in range(episode_length):
                _add_frame(
                    dataset,
                    {
                        "observation.images.head_rgb": rgb_head_resized[index],
                        "observation.images.left_wrist_rgb": rgb_left_hand_resized[index],
                        "observation.images.right_wrist_rgb": rgb_right_hand_resized[index],
                        "observation.state.left_arm_joints": state[index, :6],
                        "observation.state.left_gripper": state[index, 6:7],
                        "observation.state.right_arm_joints": state[index, 7:13],
                        "observation.state.right_gripper": state[index, 13:14],
                        "action.left_arm_joints": action[index, :6],
                        "action.left_gripper": action[index, 6:7],
                        "action.right_arm_joints": action[index, 7:13],
                        "action.right_gripper": action[index, 13:14],
                    },
                    episode_instruction,
                )
        dataset.save_episode()

    print(f"Dataset {task} created successfully at {output_root} with {len(dataset)} frames.")
    if push_to_hub:
        raise NotImplementedError("Pushing to the Hugging Face Hub is not supported yet.")


if __name__ == "__main__":
    tyro.cli(main)
