import h5py
import numpy as np

from galaxea_sim.utils.data_utils import (
    PRE_ACTION_RECORDING_CONTRACT,
    record_observation_action,
    save_dict_list_to_hdf5,
    validate_recording_contract,
)


def _observation() -> dict:
    return {
        "upper_body_observations": {
            "left_arm_joint_position": np.zeros(6, dtype=np.float32),
            "right_arm_joint_position": np.zeros(6, dtype=np.float32),
            "left_arm_ee_pose": np.zeros(7, dtype=np.float32),
            "right_arm_ee_pose": np.zeros(7, dtype=np.float32),
        },
        "upper_body_action_dict": {
            "left_arm_joint_position_cmd": np.zeros(6, dtype=np.float32),
            "left_arm_gripper_position_cmd": np.zeros(1, dtype=np.float32),
            "right_arm_joint_position_cmd": np.zeros(6, dtype=np.float32),
            "right_arm_gripper_position_cmd": np.zeros(1, dtype=np.float32),
            "left_arm_ee_pose_cmd": np.zeros(7, dtype=np.float32),
            "right_arm_ee_pose_cmd": np.zeros(7, dtype=np.float32),
        },
    }


def test_record_observation_action_pairs_pre_step_state_with_action() -> None:
    observation = _observation()
    action = np.asarray([*range(6), 0.03, *range(6, 12), 0.04], dtype=np.float32)

    recorded = record_observation_action(observation, action)

    np.testing.assert_array_equal(
        recorded["upper_body_observations"]["left_arm_joint_position"],
        observation["upper_body_observations"]["left_arm_joint_position"],
    )
    np.testing.assert_array_equal(
        recorded["upper_body_action_dict"]["left_arm_joint_position_cmd"], action[:6]
    )
    np.testing.assert_array_equal(
        recorded["upper_body_action_dict"]["right_arm_joint_position_cmd"], action[7:13]
    )
    np.testing.assert_array_equal(
        recorded["upper_body_action_dict"]["right_arm_gripper_position_cmd"],
        np.asarray([0.04], dtype=np.float32),
    )
    # The recorder must not mutate the observation that will be stepped.
    np.testing.assert_array_equal(
        observation["upper_body_action_dict"]["left_arm_joint_position_cmd"], np.zeros(6)
    )


def test_save_hdf5_stores_recording_contract(tmp_path) -> None:
    path = tmp_path / "demo.h5"
    save_dict_list_to_hdf5(
        [_observation()],
        path,
        metadata={
            "recording_contract": PRE_ACTION_RECORDING_CONTRACT,
            "controller_type": "bimanual_joint_position",
            "control_freq": 15,
            "camera_resolution_scale": 4,
        },
    )
    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["recording_contract"] == PRE_ACTION_RECORDING_CONTRACT
        assert h5_file.attrs["controller_type"] == "bimanual_joint_position"
        validate_recording_contract(
            h5_file.attrs,
            expected_controller_types={"bimanual_joint_position"},
            expected_control_freq=15,
            expected_camera_resolution_scale=4,
        )


def test_recording_contract_rejects_wrong_frequency(tmp_path) -> None:
    path = tmp_path / "demo.h5"
    save_dict_list_to_hdf5(
        [_observation()],
        path,
        metadata={
            "recording_contract": PRE_ACTION_RECORDING_CONTRACT,
            "controller_type": "bimanual_joint_position",
            "control_freq": 30,
            "camera_resolution_scale": 4,
        },
    )
    with h5py.File(path, "r") as h5_file:
        try:
            validate_recording_contract(h5_file.attrs, expected_control_freq=15)
        except ValueError as error:
            assert "frequency" in str(error)
        else:  # pragma: no cover - assertion is the test
            raise AssertionError("a 30 Hz episode must not pass the 15 Hz contract")
