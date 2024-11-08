"""Controller Interface.
Controller interface can unify MPC, and Learning-based controllers.
"""

import torch
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation as R
from lgmcts.utils import tf6d_to_mat44, mat44_to_tf6d


class Controller:
    def __init__(self, env_idx: int = 0, **kwargs):
        seed = kwargs.get("seed", 0)
        decimation = kwargs.get("decimation", 1)
        self.rng = np.random.RandomState(seed)
        self.decimation = decimation
        self.env_idx = env_idx
        self.default_params = {}
        self.param_noise = {}

    def __call__(self, obs: dict = {}, sub_goal: dict = {}, ref: dict = {}) -> Any:
        raise NotImplementedError

    def reset(self):
        pass

    def randomize_params(self):
        # randomize params
        for key in self.default_params.keys():
            self.__setattr__(key, self.default_params[key] + self.rng.uniform(-self.param_noise[key], self.param_noise[key]))
            # make sure type is correct
            if isinstance(self.default_params[key], int):
                self.__setattr__(key, int(self.__getattribute__(key)))


class ReplayController(Controller):
    """Controller executes replay control."""

    def __init__(self, env_idx: int = 0, **kwargs):
        super().__init__(env_idx=env_idx, **kwargs)
        self.is_gripper_only = True
        self.progress = 0

    def __call__(self, obs: dict = {}, sub_goal: dict = {}, ref: dict = {}) -> Any:
        trajectory = ref["trajectory"]

        if self.progress >= (len(trajectory) - 1):
            sub_goal_reached = True
        else:
            sub_goal_reached = False
        self.progress = min(self.progress + 1, len(trajectory) - 1)
        control = {}
        control["ee_pose"] = tf6d_to_mat44(trajectory[self.progress][:6])
        control["grasp_width"] = trajectory[self.progress][6]
        control["sub_goal_reached"] = sub_goal_reached
        if self.is_gripper_only and trajectory[self.progress].shape[0] == 7:
            # pad action to 8
            full_control = np.concatenate([trajectory[self.progress], trajectory[self.progress][6].reshape(1)], axis=0)
        else:
            full_control = trajectory[self.progress]
        control["control"] = full_control
        return control

    def reset(self):
        self.progress = 0
        self.randomize_params()


class TrajController(Controller):
    """Controller executes trajectory tracking control."""

    def __init__(self, env_idx: int = 0, **kwargs):
        super().__init__(env_idx=env_idx, **kwargs)
        self.is_gripper_only = True
        self.progress = 0
        self.sub_progress = 0
        self.stuck_counter = 0  # counter for stuck
        # self.max_sub_progress = int(1200 / (self.decimation + 1.0))
        self.max_sub_progress = 1
        self.max_stuck = 2
        self.trans_tolerance = 0.02
        self.vel_toler = 0.02  # 0.03
        self.rot_tolerance = 0.2
        self.default_params = {"max_sub_progress": 1, "trans_tolerance": 0.02, "rot_tolerance": 0.2, "vel_toler": 0.03, "max_stuck": 2}
        self.param_noise = {"max_sub_progress": 0, "trans_tolerance": 0.005, "rot_tolerance": 0.05, "vel_toler": 0.0, "max_stuck": 0}

    def __call__(self, obs: dict = {}, sub_goal: dict = {}, ref: dict = {}) -> Any:
        if self.is_gripper_only:
            cur_pose = obs["ee_pose"][self.env_idx][:7]
        else:
            raise NotImplementedError
        if isinstance(cur_pose, torch.Tensor):
            cur_pose = cur_pose.cpu().numpy()
        trajectory = ref["trajectory"]
        progress_goal = trajectory[self.progress]
        trans_error, rot_error, gripper_error = self._pose_error(cur_pose, progress_goal)
        # velocity
        vel_norm = torch.linalg.norm(obs["qvel"])
        # print(f"vel_norm: {vel_norm}")
        if vel_norm < self.vel_toler:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if (
            (trans_error < self.trans_tolerance and rot_error < self.rot_tolerance)
            or self.sub_progress >= self.max_sub_progress
            or self.stuck_counter >= self.max_stuck
        ):
            self.progress = min(self.progress + 1, len(trajectory) - 1)
            self.sub_progress = 0

        self.sub_progress += 1
        sub_goal_pose = sub_goal["ee_pose"]
        sub_goal_trans_error, sub_goal_rot_error, sub_goal_gripper_error = self._pose_error(cur_pose, sub_goal_pose)
        if sub_goal_trans_error < self.trans_tolerance and sub_goal_rot_error < self.rot_tolerance and self.progress >= len(trajectory) - 1:
            sub_goal_reached = True
        else:
            sub_goal_reached = False

        control = {}
        control["ee_pose"] = tf6d_to_mat44(trajectory[self.progress][:6])
        control["grasp_width"] = trajectory[self.progress][6]
        control["sub_goal_reached"] = sub_goal_reached
        if self.is_gripper_only and trajectory[self.progress].shape[0] == 7:
            # pad action to 8
            full_control = np.concatenate([trajectory[self.progress], trajectory[self.progress][6].reshape(1)], axis=0)
        else:
            full_control = trajectory[self.progress]
        control["control"] = full_control
        return control

    def reset(self):
        self.progress = 0
        self.randomize_params()

    def _pose_error(self, cur_pose, goal_pose):
        trans_error = np.linalg.norm(cur_pose[:3] - goal_pose[:3])
        rot_error = np.linalg.norm((R.from_rotvec(cur_pose[3:6]).inv() * R.from_rotvec(goal_pose[3:6])).as_rotvec())
        gripper_error = np.linalg.norm(cur_pose[6] - goal_pose[6])
        return trans_error, rot_error, gripper_error


class MPCController(TrajController):
    """Controller executes MPC control.
    Consider the naive control for manipulating joint.
    """

    def __init__(self, env_idx: int = 0, **kwargs):
        super().__init__(env_idx=env_idx, **kwargs)


class LearnController(Controller):
    """Controller executes learning-based control."""

    def __init__(self, model, env_idx: int = 0, **kwargs):
        super().__init__(env_idx=env_idx, **kwargs)
        self.model = model

    def __call__(self, obs: dict = {}, sub_goal: dict = {}, ref: dict = {}) -> Any:
        raise NotImplementedError
