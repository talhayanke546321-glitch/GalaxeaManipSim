import re
import shutil

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from typing import Literal
import tyro
import glob
import h5py
import numpy as np

from galaxea_sim.utils.data_utils import (
    IMAGE_PREPROCESS_CONTRACT,
    PRE_ACTION_RECORDING_CONTRACT,
    validate_recording_contract,
)
from galaxea_sim.utils.image_utils import prepare_rgb_image

REPO_PREFIX = "galaxea"
_TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_task_name(task: str) -> None:
    if not _TASK_NAME_PATTERN.fullmatch(task):
        raise ValueError(
            "task names must be a single path-safe component containing only "
            "letters, numbers, '.', '_' or '-': "
            f"{task!r}"
        )


def _decode_instruction(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _episode_instruction(h5_file: h5py.File, fallback: str) -> str:
    if "language_instruction" not in h5_file:
        return fallback
    values = h5_file["language_instruction"][()]
    if len(values) == 0:
        return fallback
    instruction = _decode_instruction(values[0])
    if any(_decode_instruction(value) != instruction for value in values[1:]):
        raise ValueError("language_instruction must be constant within one episode")
    return instruction


def main(
    task: str,
    data_dir: str = "datasets",
    tag: str | None = None,
    robot: Literal["r1", "r1_pro", "r1_lite"] = "r1",
    use_eef: bool = False,
    use_video: bool = False,
    push_to_hub: bool = False,
):
    _validate_task_name(task)
    raw_dataset_name = task
    if tag:
        h5_paths = sorted(glob.glob(f"{data_dir}/{raw_dataset_name}/{tag}/*.h5", recursive=True))
    else:
        h5_paths = sorted(glob.glob(f"{data_dir}/{raw_dataset_name}/**/*.h5", recursive=True))
    if not h5_paths:
        raise FileNotFoundError(f"No HDF5 episodes found for task {raw_dataset_name!r}")
    expected_controllers = {"bimanual_ee_pose", "bimanual_relaxed_ik"} if use_eef else {"bimanual_joint_position"}
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5_file:
            validate_recording_contract(
                h5_file.attrs,
                expected_controller_types=expected_controllers,
                expected_control_freq=15,
                expected_camera_resolution_scale=4,
            )

    output_path = HF_LEROBOT_HOME / REPO_PREFIX / task
    if output_path.exists():
        shutil.rmtree(output_path)
    shape = (224, 224, 3)  # Resize images to 224x224
    arm_dof = 7 if (robot == 'r1_pro' or use_eef) else 6
    
    dataset = LeRobotDataset.create(
        repo_id=f"{REPO_PREFIX}/{output_path.name}",
        robot_type=robot,
        fps=15,
        features={
            "observation.images.rgb_head": {
                "dtype": "image" if not use_video else "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.rgb_left_hand": {
                "dtype": "image" if not use_video else "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.rgb_right_hand": {
                "dtype": "image" if not use_video else "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (arm_dof * 2 + 2,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (arm_dof * 2 + 2,),
                "names": None,
            },
        },
        use_videos=use_video,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    dataset.meta.info.update(
        {
            "galaxea_recording_contract": PRE_ACTION_RECORDING_CONTRACT,
            "galaxea_controller_type": (
                "bimanual_ee_pose" if use_eef else "bimanual_joint_position"
            ),
            "galaxea_image_preprocess": IMAGE_PREPROCESS_CONTRACT,
            "galaxea_robot": robot,
            "galaxea_action_dim": arm_dof * 2 + 2,
            "galaxea_control_freq": 15,
            "galaxea_camera_resolution_scale": 4,
        }
    )
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            episode_instruction = _episode_instruction(f, fallback=raw_dataset_name)
            rgb_head = f["upper_body_observations"]["rgb_head"][()]
            rgb_left_hand = f["upper_body_observations"]["rgb_left_hand"][()]
            rgb_right_hand = f["upper_body_observations"]["rgb_right_hand"][()]

            rgb_head_resized = np.stack(
                [
                    prepare_rgb_image(img, shape[0], shape[1], name="rgb_head")
                    for img in rgb_head
                ]
            )
            rgb_left_hand_resized = np.stack(
                [
                    prepare_rgb_image(img, shape[0], shape[1], name="rgb_left_hand")
                    for img in rgb_left_hand
                ]
            )
            rgb_right_hand_resized = np.stack(
                [
                    prepare_rgb_image(img, shape[0], shape[1], name="rgb_right_hand")
                    for img in rgb_right_hand
                ]
            )

            left_arm_joint_position = f["upper_body_observations"][
                "left_arm_joint_position"
            ][()]
            left_arm_gripper_position = f["upper_body_observations"][
                "left_arm_gripper_position"
            ][()]
            right_arm_joint_position = f["upper_body_observations"][
                "right_arm_joint_position"
            ][()]
            right_arm_gripper_position = f["upper_body_observations"][
                "right_arm_gripper_position"
            ][()]

            if use_eef:
                left_arm_ee_pose = f["upper_body_observations"]["left_arm_ee_pose"][()]
                right_arm_ee_pose = f["upper_body_observations"]["right_arm_ee_pose"][()]

            left_arm_state, right_arm_state = (
                (left_arm_ee_pose, right_arm_ee_pose)
                if use_eef
                else (left_arm_joint_position, right_arm_joint_position)
            )
            state = np.concatenate(
                [
                    left_arm_state,
                    left_arm_gripper_position,
                    right_arm_state,
                    right_arm_gripper_position,
                ],
                axis=-1,
            )

            left_arm_joint_position_cmd = f["upper_body_action_dict"][
                "left_arm_joint_position_cmd"
            ][()]
            left_arm_gripper_position_cmd = f["upper_body_action_dict"][
                "left_arm_gripper_position_cmd"
            ][()]
            right_arm_joint_position_cmd = f["upper_body_action_dict"][
                "right_arm_joint_position_cmd"
            ][()]
            right_arm_gripper_position_cmd = f["upper_body_action_dict"][
                "right_arm_gripper_position_cmd"
            ][()]

            if use_eef:
                left_arm_ee_pose_cmd = f["upper_body_action_dict"]["left_arm_ee_pose_cmd"][()]
                right_arm_ee_pose_cmd = f["upper_body_action_dict"]["right_arm_ee_pose_cmd"][()]

            left_arm_action, right_arm_action = (
                (left_arm_ee_pose_cmd, right_arm_ee_pose_cmd)
                if use_eef
                else (left_arm_joint_position_cmd, right_arm_joint_position_cmd)
            )
            action = np.concatenate(
                [
                    left_arm_action,
                    left_arm_gripper_position_cmd,
                    right_arm_action,
                    right_arm_gripper_position_cmd,
                ],
                axis=-1,
            )

            episode_length = rgb_head.shape[0]
            if not (
                len(rgb_left_hand) == episode_length
                and len(rgb_right_hand) == episode_length
                and len(state) == episode_length
                and len(action) == episode_length
            ):
                raise ValueError(f"Episode arrays have inconsistent lengths in {h5_path}")
            for i in range(episode_length):
                dataset.add_frame(
                    {
                        "observation.images.rgb_head": rgb_head_resized[i],
                        "observation.images.rgb_left_hand": rgb_left_hand_resized[i],
                        "observation.images.rgb_right_hand": rgb_right_hand_resized[i],
                        "observation.state": state[i].astype(np.float32),
                        "action": action[i].astype(np.float32),
                    },
                    task=episode_instruction,
                )
        dataset.save_episode()
    print(f"Dataset {output_path.name} created successfully with {len(dataset)} frames.")
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        raise NotImplementedError("Pushing to the Hugging Face Hub is not supported yet.")


if __name__ == "__main__":
    tyro.cli(main)
