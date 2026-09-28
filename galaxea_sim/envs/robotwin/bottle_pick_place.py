"""R1 Pro single-bottle pick-and-place task.

The existing ``R1ProDualBottlesPickEasy`` task is useful for reachability, but it
only asks the robot to lift two bottles to floating goal poses.  This task keeps
the same bottle asset and right-arm grasp geometry while adding a physical plate
on the table and a release/stability check.  It is intentionally deterministic
apart from the bottle's XY start pose so the 100 demonstration collection can be
stratified by table height without changing the action/observation contract.
"""

from copy import deepcopy

import numpy as np
import sapien
from sapien import Pose

from galaxea_sim.utils.robotwin_utils import create_glb, rand_create_glb

from .robotwin_base import RoboTwinBaseEnv


class BottlePickPlaceEnv(RoboTwinBaseEnv):
    """Pick the bottle on the right side and release it on the plate."""

    bottle_model_id = 13
    bottle_scale = (0.132, 0.132, 0.132)
    bottle_qpos = [0.707, 0.707, 0.0, 0.0]

    @property
    def table_height(self):
        if self._table_height_override is not None:
            return self._table_height_override
        return 0.9

    @property
    def tabletop_center_x(self):
        return 0.7

    def _setup_plate(self):
        # Coordinates passed to create_glb are relative to tabletop_center_in_world.
        self.plate, _ = create_glb(
            self._scene,
            pose=Pose(
                [
                    self.tabletop_center_in_world[0] + self._target_xy[0],
                    self.tabletop_center_in_world[1] + self._target_xy[1],
                    self.table_height + 0.013,
                ],
                [0.5, 0.5, 0.5, 0.5],
            ),
            modelname="003_plate",
            scale=(0.025, 0.025, 0.025),
            is_static=True,
            convex=True,
        )

    def _setup_bottle(self):
        self.bottle, self.bottle_data = rand_create_glb(
            scene=self._scene,
            modelname="001_bottles",
            xlim=[-0.22, -0.08],
            ylim=[-0.25, -0.10],
            zlim=[0.125],
            rotate_rand=False,
            qpos=self.bottle_qpos,
            scale=self.bottle_scale,
            model_id=self.bottle_model_id,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            model_z_val=True,
            convex=True,
        )
        self.bottle_initial_pose_on_table = self.bottle.get_pose()

    def reset_world(self, reset_info=None):
        if hasattr(self, "plate"):
            self._scene.remove_actor(self.plate)
        if hasattr(self, "bottle"):
            self._scene.remove_actor(self.bottle)

        # Keep target coordinates fixed in table-relative coordinates so changing
        # table height only changes the vertical reach, not the image layout.
        self._target_xy = np.array([-0.05, -0.16], dtype=np.float32)
        self._setup_plate()
        self._setup_bottle()
        if reset_info is not None:
            self.bottle.set_pose(
                Pose(
                    p=reset_info["bottle_initial_pose"][:3],
                    q=reset_info["bottle_initial_pose"][3:],
                )
            )
            self.bottle_initial_pose_on_table = self.bottle.get_pose()

        self._placement_hold_steps = 0
        self._released = False

    @property
    def target_bottle_pose(self):
        return Pose(
            p=np.array(
                [
                    self.tabletop_center_in_world[0] + self._target_xy[0],
                    self.tabletop_center_in_world[1] + self._target_xy[1],
                    self.table_height + 0.125,
                ],
                dtype=np.float32,
            ),
            q=np.asarray(self.bottle_qpos, dtype=np.float32),
        )

    def _right_grasp_poses(self):
        # This is the validated R1 Pro bottle grasp used by the existing dual
        # bottle task.  The same constant offset is retained for the place pose,
        # so the bottle orientation is not changed while it is carried.
        grasp_orientation = Pose()
        grasp_orientation.set_rpy(
            rpy=np.array([0.0, 0.0, 0.88], dtype=np.float32)
            + self.robot.right_ee_rpy_offset
        )
        bottle_position = self.bottle.get_pose().p
        approach = Pose(
            p=bottle_position + np.array([-0.1096, -0.1164, 0.08]),
            q=grasp_orientation.q,
        )
        grasp = Pose(
            p=bottle_position + np.array([-0.0196, -0.0164, 0.0]),
            q=grasp_orientation.q,
        )
        target_bottle = self.target_bottle_pose
        carry_offset = grasp.p - bottle_position
        place = Pose(p=target_bottle.p + carry_offset, q=grasp.q)
        above_place = Pose(p=place.p + np.array([0.0, 0.0, 0.14]), q=place.q)
        retreat = Pose(p=above_place.p + np.array([0.0, 0.0, 0.10]), q=above_place.q)
        return approach, grasp, above_place, place, retreat

    def solution(self):
        approach, grasp, above_place, place, retreat = self._right_grasp_poses()
        yield ("move_to_pose", {"right_pose": deepcopy(approach)})
        yield ("open_gripper", {"action_mode": "right", "gripper_target_state": 0.05})
        yield ("move_to_pose", {"right_pose": deepcopy(grasp)})
        yield ("close_gripper", {"action_mode": "right"})
        yield ("move_to_pose", {"right_pose": deepcopy(above_place)})
        yield ("move_to_pose", {"right_pose": deepcopy(place)})
        yield ("open_gripper", {"action_mode": "right", "gripper_target_state": 0.05})
        yield ("move_to_pose", {"right_pose": deepcopy(retreat)})

    def _placement_metrics(self):
        bottle_position = self.bottle.get_pose().p
        target_position = self.target_bottle_pose.p
        delta = bottle_position - target_position
        xy_error = float(np.linalg.norm(delta[:2]))
        z_error = float(abs(delta[2]))
        # The bottle's long axis is local +Y after ``bottle_qpos`` rotates the
        # mesh upright.  Measure its deviation from world Z instead of comparing
        # the full quaternion (yaw around the vertical axis is harmless).
        long_axis = self.bottle.get_pose().to_transformation_matrix()[:3, 1]
        tilt_deg = float(
            np.degrees(np.arccos(np.clip(abs(float(long_axis[2])), 0.0, 1.0)))
        )
        placed = xy_error < 0.055 and z_error < 0.035 and tilt_deg < 35.0
        return xy_error, z_error, tilt_deg, placed

    def _get_info(self):
        xy_error, z_error, tilt_deg, placed = self._placement_metrics()
        success = bool(placed and self._released and self._placement_hold_steps >= 10)
        return dict(
            success=success,
            placed=bool(placed),
            released=bool(self._released),
            placement_hold_steps=int(self._placement_hold_steps),
            placement_xy_error=xy_error,
            placement_z_error=z_error,
            placement_tilt_deg=tilt_deg,
            table_height_m=float(self.table_height),
        )

    def _get_reset_info(self):
        return dict(
            bottle_initial_pose=np.concatenate(
                [self.bottle_initial_pose_on_table.p, self.bottle_initial_pose_on_table.q]
            )
        )

    def _check_termination(self) -> bool:
        if self.last_gripper_cmd[1] >= 0.045:
            self._released = True
        _, _, _, placed = self._placement_metrics()
        if self._released and placed:
            self._placement_hold_steps += 1
        else:
            self._placement_hold_steps = 0
        return self._placement_hold_steps >= 10

    @property
    def language_instruction(self):
        return "pick up the bottle and place it on the plate"

    def get_object_dict(self):
        return dict(
            plate=np.concatenate([self.plate.get_pose().p, self.plate.get_pose().q]),
            bottle=np.concatenate([self.bottle.get_pose().p, self.bottle.get_pose().q]),
            target_bottle=np.concatenate(
                [self.target_bottle_pose.p, self.target_bottle_pose.q]
            ),
        )

    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0
