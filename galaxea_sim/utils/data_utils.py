"""仿真轨迹的 HDF5/JSON 存储和时序契约工具。

训练数据最容易出现的隐蔽错误是时间错位：把执行动作之后的观测和这个
动作放在同一帧，会把真实的 ``(state_t, action_t)`` 变成错误的
``(state_{t+1}, action_t)``。本文件通过 ``pre_action_v1`` 明确规定：
记录动作前的 observation，并在推进环境之前把即将执行的 action 写入
对应 command 字段。
"""

import copy
import h5py
import json
import numpy as np
import os


PRE_ACTION_RECORDING_CONTRACT = "pre_action_v1"
IMAGE_PREPROCESS_CONTRACT = "resize_with_pad_pil_bilinear_v1"


def validate_recording_contract(
    attrs: object,
    *,
    expected_controller_types: set[str] | None = None,
    expected_control_freq: int | None = None,
    expected_camera_resolution_scale: int | None = None,
) -> None:
    """校验 HDF5 轨迹的时序、控制器、频率和相机版本契约。

    旧轨迹如果没有明确标记，就无法证明 state/action 是否对齐，因此这里
    选择 fail closed：拒绝转换，而不是静默猜测后产生训练污染。
    """

    def attribute(name: str):
        value = attrs.get(name) if hasattr(attrs, "get") else None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8")
        return value

    recording_contract = attribute("recording_contract")
    if recording_contract != PRE_ACTION_RECORDING_CONTRACT:
        raise ValueError(
            "HDF5 episode has no supported pre-action recording contract; "
            "recollect/replay it with the current simulator scripts before conversion"
        )

    if expected_controller_types is not None:
        controller_type = attribute("controller_type")
        if controller_type not in expected_controller_types:
            raise ValueError(
                f"HDF5 controller_type={controller_type!r} is incompatible; "
                f"expected one of {sorted(expected_controller_types)}"
            )

    if expected_control_freq is not None:
        actual_control_freq = attribute("control_freq")
        try:
            frequency_matches = int(actual_control_freq) == expected_control_freq
        except (TypeError, ValueError):
            frequency_matches = False
        if not frequency_matches:
            raise ValueError(
                "HDF5 control frequency is incompatible: "
                f"got {actual_control_freq!r}, expected {expected_control_freq} Hz"
            )

    if expected_camera_resolution_scale is not None:
        actual_scale = attribute("camera_resolution_scale")
        try:
            scale_matches = int(actual_scale) == expected_camera_resolution_scale
        except (TypeError, ValueError):
            scale_matches = False
        if not scale_matches:
            raise ValueError(
                "HDF5 camera resolution scale is incompatible: "
                f"got {actual_scale!r}, expected {expected_camera_resolution_scale}"
            )


def record_observation_action(
    observation: dict,
    action: np.ndarray,
    *,
    controller_type: str = "bimanual_joint_position",
) -> dict:
    """把动作前观测和即将执行的动作配成一帧。

    Pair a pre-step observation with the action about to be executed.

    ``Env.step`` returns the observation *after* applying an action.  Storing
    that returned observation beside the same action shifts behavior-cloning
    samples by one control step.  This helper copies the current observation
    and stamps the action into the matching command fields before the caller
    advances the simulator.
    """

    recorded = copy.deepcopy(observation)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(action)):
        raise ValueError("action must contain only finite values")
    upper_observations = recorded.get("upper_body_observations", {})
    command = recorded.get("upper_body_action_dict")
    if not isinstance(command, dict) or not isinstance(upper_observations, dict):
        raise ValueError("observation must contain upper-body observations and action dictionaries")

    if controller_type == "bimanual_joint_position":
        left_dim = np.asarray(upper_observations["left_arm_joint_position"]).size
        right_dim = np.asarray(upper_observations["right_arm_joint_position"]).size
        expected_dim = left_dim + 1 + right_dim + 1
        if action.size != expected_dim:
            raise ValueError(f"joint-position action must have {expected_dim} values, got {action.shape}")
        command["left_arm_joint_position_cmd"] = action[:left_dim].copy()
        command["left_arm_gripper_position_cmd"] = np.asarray([action[left_dim]], dtype=np.float32)
        right_start = left_dim + 1
        command["right_arm_joint_position_cmd"] = action[right_start : right_start + right_dim].copy()
        command["right_arm_gripper_position_cmd"] = np.asarray([action[-1]], dtype=np.float32)
    elif controller_type in {"bimanual_ee_pose", "bimanual_relaxed_ik"}:
        left_dim = np.asarray(upper_observations["left_arm_ee_pose"]).size
        right_dim = np.asarray(upper_observations["right_arm_ee_pose"]).size
        expected_dim = left_dim + 1 + right_dim + 1
        if action.size != expected_dim:
            raise ValueError(f"EEF action must have {expected_dim} values, got {action.shape}")
        command["left_arm_ee_pose_cmd"] = action[:left_dim].copy()
        command["left_arm_gripper_position_cmd"] = np.asarray([action[left_dim]], dtype=np.float32)
        right_start = left_dim + 1
        command["right_arm_ee_pose_cmd"] = action[right_start : right_start + right_dim].copy()
        command["right_arm_gripper_position_cmd"] = np.asarray([action[-1]], dtype=np.float32)
    else:
        raise ValueError(f"unsupported controller type: {controller_type}")

    return recorded

def recursive_create_dict_list(d):
    """按第一帧字典结构创建同形状的 Python list 容器。"""
    if isinstance(d, dict):
        return {k: recursive_create_dict_list(v) for k, v in d.items()}
    else:
        return []
    
def recursive_append_dict_list(f, d, data):
    """递归把一帧数据追加到 HDF5 写入缓冲区。"""
    if isinstance(data, dict):
        for k, v in data.items():
            recursive_append_dict_list(f, d[k], v)
    else:
        if isinstance(data, str):
            # d.append(data)
            # Store UTF-8 so language instructions are not limited to ASCII.
            d.append(data.encode("utf-8"))
        else:
            d.append(data)
        
def recursive_convert_dict_list_to_numpy(d):
    """把递归 list 结构转换成 NumPy 数组结构。"""
    if isinstance(d, dict):
        return {k: recursive_convert_dict_list_to_numpy(v) for k, v in d.items()}
    else:
        return np.array(d)

def recursive_create_dict_list_to_hdf5(f, d, key=""):
    """递归创建 HDF5 dataset，并统一使用 gzip 压缩。"""
    if isinstance(d, dict):
        for k, v in d.items():
            recursive_create_dict_list_to_hdf5(f, v, key + "/" + k)
    else:
        f.create_dataset(key, data=d, compression="gzip")

def save_dict_list_to_hdf5(dict_list, file_path, metadata=None):
    """将一条轨迹和其契约元数据写入 HDF5 文件。"""
    if not dict_list:
        raise ValueError("cannot save an empty trajectory")
    # overwrite the file
    if os.path.exists(file_path):
        os.remove(file_path)
    parent_dir = os.path.dirname(os.fspath(file_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        if metadata:
            for key, value in metadata.items():
                f.attrs[str(key)] = value
        output_dict = recursive_create_dict_list(dict_list[0])
        for i, d in enumerate(dict_list):
            recursive_append_dict_list(f, output_dict, d)
        output_dict_numpy = recursive_convert_dict_list_to_numpy(output_dict)
        recursive_create_dict_list_to_hdf5(f, output_dict_numpy, "")
        
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, o):
        # Check for NumPy integer types
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(o)
            
        # Check for NumPy float types
        elif isinstance(o, (np.float16, np.float32, np.float64)):
            return float(o)
        
        # ADD THIS BLOCK to handle NumPy booleans
        elif isinstance(o, np.bool_):
            return bool(o)
            
        # Check for NumPy arrays
        elif isinstance(o, np.ndarray):
            return o.tolist()
            
        return super(NumpyEncoder, self).default(o)
        
def save_dict_list_to_json(dict_list, file_path):
    """使用 NumPy 兼容编码器保存 reset/采集统计信息。"""
    parent_dir = os.path.dirname(os.fspath(file_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(dict_list, f, cls=NumpyEncoder, indent=4)
