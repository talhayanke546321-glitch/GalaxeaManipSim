"""ROS2 transport for the Galaxea R1 Pro real-robot OpenPI client.

``rclpy`` is imported lazily so protocol/unit tests can run on the GPU
workstation without a ROS installation.  Constructing :class:`R1ProRos2Bridge`
on a machine without ROS2 Humble fails with an actionable error.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

import cv2
import numpy as np

from .r1_pro_real_adapter import R1ProHardwareCommand, R1ProHardwareSnapshot, SNAPSHOT_KEYS


@dataclasses.dataclass(frozen=True)
class R1ProRosTopics:
    """Official R1 Pro ROS2 feedback, camera and motion-target topics."""

    left_arm_state: str = "/hdas/feedback_arm_left"
    right_arm_state: str = "/hdas/feedback_arm_right"
    left_gripper_state: str = "/hdas/feedback_gripper_left"
    right_gripper_state: str = "/hdas/feedback_gripper_right"
    head_camera: str = "/hdas/camera_head/left_raw/image_raw_color/compressed"
    left_wrist_camera: str = "/hdas/camera_wrist_left/color/image_raw/compressed"
    right_wrist_camera: str = "/hdas/camera_wrist_right/color/image_raw/compressed"
    left_arm_command: str = "/motion_target/target_joint_state_arm_left"
    right_arm_command: str = "/motion_target/target_joint_state_arm_right"
    left_gripper_command: str = "/motion_target/target_position_gripper_left"
    right_gripper_command: str = "/motion_target/target_position_gripper_right"


class R1ProRos2Bridge:
    """Collect synchronized observations and optionally publish guarded commands."""

    def __init__(
        self,
        *,
        enable_actuation: bool = False,
        arm_velocity_limit: float = 0.25,
        topics: R1ProRosTopics | None = None,
        node_name: str = "openpi_r1_pro_real_client",
    ) -> None:
        if arm_velocity_limit <= 0:
            raise ValueError("arm_velocity_limit must be positive")
        self.enable_actuation = bool(enable_actuation)
        self.arm_velocity_limit = float(arm_velocity_limit)
        self.topics = topics or R1ProRosTopics()
        self._samples: dict[str, tuple[np.ndarray, float]] = {}
        self._condition = threading.Condition()
        self._closed = False

        try:
            import rclpy
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import CompressedImage, JointState
        except ImportError as exc:  # pragma: no cover - exercised on the robot host
            raise RuntimeError(
                "ROS2 Python packages are unavailable. Source /opt/ros/humble/setup.bash "
                "and the Galaxea SDK install/setup.bash before starting the real client."
            ) from exc

        self._rclpy = rclpy
        self._joint_state_type = JointState
        self._owns_ros_context = not rclpy.ok()
        if self._owns_ros_context:
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        callback_group = ReentrantCallbackGroup()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._subscriptions = [
            self.node.create_subscription(
                JointState,
                self.topics.left_arm_state,
                lambda msg: self._joint_callback("left_arm", msg, 7),
                qos,
                callback_group=callback_group,
            ),
            self.node.create_subscription(
                JointState,
                self.topics.right_arm_state,
                lambda msg: self._joint_callback("right_arm", msg, 7),
                qos,
                callback_group=callback_group,
            ),
            self.node.create_subscription(
                JointState,
                self.topics.left_gripper_state,
                lambda msg: self._joint_callback("left_gripper", msg, 1),
                qos,
                callback_group=callback_group,
            ),
            self.node.create_subscription(
                JointState,
                self.topics.right_gripper_state,
                lambda msg: self._joint_callback("right_gripper", msg, 1),
                qos,
                callback_group=callback_group,
            ),
            self.node.create_subscription(
                CompressedImage,
                self.topics.head_camera,
                lambda msg: self._image_callback("head_rgb", msg),
                qos,
                callback_group=callback_group,
            ),
            self.node.create_subscription(
                CompressedImage,
                self.topics.left_wrist_camera,
                lambda msg: self._image_callback("left_wrist_rgb", msg),
                qos,
                callback_group=callback_group,
            ),
            self.node.create_subscription(
                CompressedImage,
                self.topics.right_wrist_camera,
                lambda msg: self._image_callback("right_wrist_rgb", msg),
                qos,
                callback_group=callback_group,
            ),
        ]

        self._publishers: dict[str, Any] = {}
        if self.enable_actuation:
            self._publishers = {
                "left_arm": self.node.create_publisher(JointState, self.topics.left_arm_command, qos),
                "right_arm": self.node.create_publisher(JointState, self.topics.right_arm_command, qos),
                "left_gripper": self.node.create_publisher(
                    JointState, self.topics.left_gripper_command, qos
                ),
                "right_gripper": self.node.create_publisher(
                    JointState, self.topics.right_gripper_command, qos
                ),
            }

        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="r1-pro-ros2-spin",
            daemon=True,
        )
        self._spin_thread.start()

    def _store(self, name: str, value: np.ndarray) -> None:
        with self._condition:
            self._samples[name] = (np.asarray(value).copy(), time.monotonic())
            self._condition.notify_all()

    def _joint_callback(self, name: str, message: Any, size: int) -> None:
        value = np.asarray(message.position, dtype=np.float32).reshape(-1)
        if value.size < size or not np.all(np.isfinite(value[:size])):
            self.node.get_logger().error(
                f"Ignoring invalid {name} feedback with shape {value.shape}"
            )
            return
        self._store(name, value[:size])

    def _image_callback(self, name: str, message: Any) -> None:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            self.node.get_logger().error(f"Ignoring undecodable image on {name}")
            return
        self._store(name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def missing_observations(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(name for name in SNAPSHOT_KEYS if name not in self._samples)

    def wait_until_ready(self, timeout_seconds: float = 20.0) -> None:
        """Wait until every required camera and feedback stream has produced data."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                missing = tuple(name for name in SNAPSHOT_KEYS if name not in self._samples)
                if not missing:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for R1 Pro ROS2 observations: " + ", ".join(missing)
                    )
                self._condition.wait(timeout=remaining)

    def snapshot(self) -> R1ProHardwareSnapshot:
        """Copy the newest value from each stream into an immutable snapshot."""
        with self._condition:
            missing = tuple(name for name in SNAPSHOT_KEYS if name not in self._samples)
            if missing:
                raise RuntimeError("R1 Pro observations are not ready: " + ", ".join(missing))
            values = {name: self._samples[name][0].copy() for name in SNAPSHOT_KEYS}
            receive_times = {name: self._samples[name][1] for name in SNAPSHOT_KEYS}
        return R1ProHardwareSnapshot(
            head_rgb=values["head_rgb"],
            left_wrist_rgb=values["left_wrist_rgb"],
            right_wrist_rgb=values["right_wrist_rgb"],
            left_arm=values["left_arm"],
            left_gripper=values["left_gripper"],
            right_arm=values["right_arm"],
            right_gripper=values["right_gripper"],
            receive_times=receive_times,
        )

    def _joint_message(self, positions: object, *, include_velocity: bool) -> Any:
        message = self._joint_state_type()
        message.header.stamp = self.node.get_clock().now().to_msg()
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        message.position = values.tolist()
        if include_velocity:
            message.velocity = [self.arm_velocity_limit] * values.size
        return message

    def publish(self, command: R1ProHardwareCommand) -> None:
        """Publish one already-validated hardware command."""
        if not self.enable_actuation:
            raise RuntimeError("ROS2 bridge is in shadow mode; action publishers were not created")
        self._publishers["left_arm"].publish(
            self._joint_message(command.left_arm, include_velocity=True)
        )
        self._publishers["right_arm"].publish(
            self._joint_message(command.right_arm, include_velocity=True)
        )
        self._publishers["left_gripper"].publish(
            self._joint_message([command.left_gripper], include_velocity=False)
        )
        self._publishers["right_gripper"].publish(
            self._joint_message([command.right_gripper], include_velocity=False)
        )

    def publish_hold(self, snapshot: R1ProHardwareSnapshot | None = None) -> None:
        """Command the latest measured pose once when an execution loop stops."""
        if not self.enable_actuation:
            return
        current = snapshot or self.snapshot()
        command = R1ProHardwareCommand(
            left_arm=current.left_arm,
            left_gripper=float(current.left_gripper[0]),
            right_arm=current.right_arm,
            right_gripper=float(current.right_gripper[0]),
        )
        self.publish(command)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(timeout_sec=2.0)
        self._spin_thread.join(timeout=2.0)
        self._executor.remove_node(self.node)
        self.node.destroy_node()
        if self._owns_ros_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def __enter__(self) -> "R1ProRos2Bridge":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["R1ProRos2Bridge", "R1ProRosTopics"]
