"""Motion Planner interface.
This is planning between two scene states.
"""

import numpy as np
import torch
from copy import deepcopy
from lgmcts.sampler import SceneState, Sampler, Action, Goal
from lgmcts.utils import tf6d_to_mat44, mat44_to_tf6d
from lgmcts.object_primitives import sample_poses_with_gripper
from scipy.spatial.transform import Rotation as R


########################################## Grasp pose utils ##########################################
def find_path_pick_place(
    sampler,
    obj_idx,
    goal_space_name,
    space_type="drawer",
    max_candidates=3,
    rng=np.random.RandomState(0),
    debug=False,
    mp_config={},  # Motion planner config
):
    """Currently, we divide spaces into two categories: drawer and shelf.
    For drawer, we grasp from the top and place from the top.
    For shelf, we grasp from the side and place from the side.
    """
    # Parameters
    safety_margin = mp_config.get("safety_margin", 0.04)
    ee_tip_offset = mp_config.get("ee_tip_offset_rigid", 0.09)
    bias_range = mp_config.get("bias_range", 0.1)
    if space_type == "drawer":
        strategy = "top"
        angle_divide = 2
    elif space_type == "shelf":
        strategy = "top"  # FIXME: use top for now
        angle_divide = 1
    elif space_type == "open":
        strategy = "top"
        angle_divide = 1
    else:
        raise ValueError(f"Unknown space type: {space_type}")
    # # strategy = "top"
    # # sampler.art_list[0].set_joint_values([0.0, 1.0, 0.0, 0.0], is_rel=True)
    # # FIXME: A easier way here can be directly sample the pose for both object and gripper
    # goal_poses = sampler.sample_poses(obj_idx, goal_space_name, n_samples=n_samples, enable_rot=True, rng=rng)
    # if goal_poses is None:
    #     print("[Warning] Object is too large to fit in the space.")
    #     return []
    manip_obj = deepcopy(sampler.rigid_list[obj_idx])
    grasp_poses, grasp_widths = manip_obj.get_grasp_poses(ee_tip_offset, wrt_obj=True, strategy=strategy, bias=rng.uniform(-bias_range, bias_range))
    gripper = deepcopy(sampler.robot.get_ee_link())

    valid_grasp_goal_pairs = []
    for grasp_pose, grasp_width in zip(grasp_poses, grasp_widths):
        obj_poses = sample_poses_with_gripper(
            sampler.spaces[goal_space_name],
            manip_obj.extent,
            gripper.extent + safety_margin,
            grasp_pose,
            n_samples=max_candidates,
            rng=rng,
            enable_rot=True,
            place_bottom=True,
            angle_divide=angle_divide,
            object=manip_obj,
            gripper=gripper,
            art_obj=sampler.art_list[0],
            debug=debug,
        )
        for obj_pose in obj_poses:
            valid_grasp_goal_pairs.append((grasp_pose, grasp_width, obj_pose))
            if len(valid_grasp_goal_pairs) >= max_candidates:
                break
        if len(valid_grasp_goal_pairs) >= max_candidates:
            break
    if len(valid_grasp_goal_pairs) == 0:
        print("[Warning] No valid grasp-goal pairs found.")
        return []
    return valid_grasp_goal_pairs


def find_path_manip_joint(sampler: Sampler, art_idx, joint_idx, goal_joint_value, rng=np.random.RandomState(0), debug=False, mp_config={}):
    """Find a path to manipulate articulated object."""
    # Parameters
    ee_tip_offset = mp_config.get("ee_tip_offset", 0.09)
    open_grasp_width = mp_config.get("open_grasp_width", 0.03)
    max_open_grasp_width = mp_config.get("max_open_grasp_width", 0.04)
    joint_waypoints = mp_config.get("joint_waypoints", 16)
    pre_offset_joint = mp_config.get("pre_offset_joint", 0.15)
    pre_num_waypoints = mp_config.get("pre_num_waypoints", 3)
    move_over_head = mp_config.get("move_over_head", 0.1)  # drag/pull overhead
    bias_range = mp_config.get("bias_range", 0.0)
    manip_traj, grasp_pose, grasp_width, status = sampler.art_list[art_idx].get_manip_traj(
        joint_idx=joint_idx,
        goal_joint_value=goal_joint_value,
        num_waypoints=joint_waypoints,
        offset=ee_tip_offset,
        pre_offset=pre_offset_joint,
        pre_num_waypoints=pre_num_waypoints,
        move_over_head=move_over_head,
        bias=rng.uniform(-bias_range, bias_range),
    )
    # sampler.show(ee_traj=manip_traj, show_floor=True)
    open_grasp_width = min(max(open_grasp_width, grasp_width * 1.5), max_open_grasp_width)

    # Current everything is valid. Later we can check motion planning feasiblity
    valid_grasp_goal_pairs = []
    if grasp_pose is not None:
        valid_grasp_goal_pairs.append((grasp_pose, grasp_width))
    return valid_grasp_goal_pairs


########################################## Motion Planner / Trajectory ##########################################


class MotionPlanner:
    def __init__(self, robot, sampler: Sampler, env_idx: int, **kwargs):
        seed = kwargs.get("seed", 0)
        self.rng = np.random.RandomState(seed)
        self.sampler = sampler
        self.env_idx = env_idx
        # Hyper parameters
        self.default_params = {
            "safety_margin": 0.02,
            # "ee_tip_offset": 0.08,  # old gripper
            "ee_tip_offset": 0.10,  # new gripper
            "ee_tip_offset_rigid": 0.135,  # object can be grasp shawllower
            "open_grasp_width": 0.03,
            "max_open_grasp_width": 0.04,
            "grasp_tight_ratio": 1.8,  # 1.6
            "pre_offset_joint": 0.20,
            "pre_offset_grasp": 0.20,
            "pre_offset_place": 0.5,
            "move_over_head": 0.05,
            "pre_num_waypoints": 3,
            "joint_place_stay": 2,
            "grasp_pose_stay": 4,
            "transit_stay": 2,
            "joint_waypoints": 20,
            "move_front_offset": 0.25,
            "linear_speed": 0.15,
            "angular_speed": np.pi / 6.0,
            "place_lift_offset": 0.1,
            "bias_range": 0.1,
        }
        self.param_noise = {
            "safety_margin": 0.00,
            "ee_tip_offset": 0.00,
            "ee_tip_offset_rigid": 0.00,
            "open_grasp_width": 0.0,
            "max_open_grasp_width": 0.0,
            "grasp_tight_ratio": 0,
            "pre_offset_joint": 0.04,
            "pre_offset_grasp": 0.05,
            "pre_offset_place": 0.05,
            "move_over_head": 0.0,
            "pre_num_waypoints": 0,
            "grasp_pose_stay": 0,
            "joint_place_stay": 0,
            "transit_stay": 0,
            "joint_waypoints": 0,
            "move_front_offset": 0.03,
            "linear_speed": 0.05,
            "angular_speed": np.pi / 12.0,
            "place_lift_offset": 0.0,
            "bias_range": 0.0,
        }
        self.params = deepcopy(self.default_params)

    def reset(self):
        """Randomize hyperparameters within a range."""
        for key in self.default_params.keys():
            self.params[key] = self.default_params[key] + self.rng.uniform(-self.param_noise[key], self.param_noise[key])
            # convert type
            if isinstance(self.default_params[key], int):
                self.params[key] = int(self.params[key])

    def plan(self, obs: dict, action: Action, neutral_pose, goal: Goal = None):
        raise NotImplementedError

    def _interpolate(self, cur_pose, goal_pose, pose_type="6d+grasp"):
        if pose_type == "6d+grasp":
            num_steps = self._compute_step(cur_pose, goal_pose)
            if num_steps <= 1:
                return np.array([goal_pose])
            elif num_steps == 2:
                return np.array([cur_pose, goal_pose])
            linear_traj = np.linspace(cur_pose[:3], goal_pose[:3], num_steps)  # (num_steps, 3)
            # Add final pose
            linear_traj = np.concatenate((linear_traj, goal_pose[:3].reshape(1, 3)), axis=0)
            angular_diff = R.from_rotvec(cur_pose[3:6]).inv() * R.from_rotvec(goal_pose[3:6])
            angular_dist = np.linalg.norm(angular_diff.as_rotvec())
            angular_axis = angular_diff.as_rotvec() / (angular_dist + 1e-6)
            angular_traj = []
            if angular_dist > 1e-6:
                for i in range(num_steps):
                    alpha = i / num_steps * angular_dist
                    interp_diff_rotvec = angular_axis * alpha
                    interp_rotvec = (R.from_rotvec(cur_pose[3:6]) * R.from_rotvec(interp_diff_rotvec)).as_rotvec()
                    angular_traj.append(interp_rotvec)
            else:
                angular_traj = [goal_pose[3:6]] * num_steps
            # Add final pose
            angular_traj.append(goal_pose[3:6])
            angular_traj = np.array(angular_traj)
            # Add grasp width
            grasp_traj = np.linspace(cur_pose[6:7], goal_pose[6:7], num_steps)
            grasp_traj = np.concatenate((grasp_traj, goal_pose[6:7].reshape(1, 1)), axis=0)
            return np.concatenate((linear_traj, angular_traj, grasp_traj.reshape(-1, 1)), axis=1)
        else:
            raise ValueError("Invalid pose shape.")

    def _compute_step(self, cur_pose: np.ndarray, goal_pose: np.ndarray):
        # Compute how many steps to reach the goal
        if cur_pose.shape == (6,) or cur_pose.shape == (7,):
            linear_dist = np.linalg.norm(goal_pose[:3] - cur_pose[:3])
            angular_diff = R.from_rotvec(cur_pose[3:6]).inv() * R.from_rotvec(goal_pose[3:6])
            angular_dist = np.linalg.norm(angular_diff.as_rotvec())
            linear_steps = int(np.ceil(linear_dist / self.params["linear_speed"]))
            angular_steps = int(np.ceil(angular_dist / self.params["angular_speed"]))
            if max(linear_steps, angular_steps) <= 1:
                return 2  # At least 2 steps
            return max(linear_steps, angular_steps)
        else:
            raise ValueError("Invalid pose shape.")


class TemplateMotionPlannerV2(MotionPlanner):
    """Motion planning using template method V2.
    Transit between action with a transition plane.
    Currently, we implement articulated motion planning and pick-place motion planning.
    """

    def __init__(self, robot, sampler, env_idx, **kwargs):
        super().__init__(robot, sampler, env_idx, **kwargs)

    def plan(self, obs: dict, action: Action, transit_plane_x: float, goal: Goal = None):
        self.sampler.sync_from_state(obs["isaac_state"][self.env_idx], scalar_first=True)
        cur_pose = obs["ee_pose"][self.env_idx]
        key_wps = []  # if this wp is key_wp
        wps = []  # in 7d, 6d pose + grasp
        if isinstance(cur_pose, torch.Tensor):
            cur_pose = cur_pose.cpu().numpy()
        if action.action_type == "art":
            manip_joint_name = action.manip_joint
            manip_art_name = action.manip_obj
            grasp_pose_local = action.grasp_pose
            grasp_width = action.grasp_width / self.params["grasp_tight_ratio"]
            art_idx = self.sampler.get_art_idx(manip_art_name)
            joint_idx = self.sampler.art_list[art_idx].active_joints.index(manip_joint_name)
            # Clip joint value between 0 and pi/2 (real value)
            goal_joint_type = self.sampler.art_list[art_idx].joints[manip_joint_name].joint_type
            if goal_joint_type == "revolute":
                real_joint_upper = np.pi / 2.0
                rel_joint_bound = self.sampler.art_list[art_idx].joints[manip_joint_name].limits
                rel_joint_upper = min(real_joint_upper / (rel_joint_bound[1] - rel_joint_bound[0] + 1e-6), 1.0)
                joint_waypoints = self.params["joint_waypoints"]
            else:
                rel_joint_upper = 1.0
                joint_waypoints = self.params["joint_waypoints"] // 2  # Half waypoints for prismatic joint
            goal_joint_value = action.goal_pose[6 + joint_idx]
            goal_joint_value = np.clip(goal_joint_value, 0.0, rel_joint_upper)
            # Generate trajectory
            manip_traj, __, _, status = self.sampler.art_list[art_idx].get_manip_traj(
                joint_idx=joint_idx,
                goal_joint_value=goal_joint_value,
                num_waypoints=joint_waypoints,
                offset=self.params["ee_tip_offset"],
                pre_offset=self.params["pre_offset_joint"],
                pre_num_waypoints=self.params["pre_num_waypoints"],
                move_over_head=self.params["move_over_head"],
                bias=self.rng.uniform(-self.params["bias_range"], self.params["bias_range"]),
                grasp_pose=grasp_pose_local,
                grasp_width=grasp_width,
            )
            # [DEBUG]
            # self.sampler.show(ee_traj=manip_traj, show_floor=True)
            open_grasp_width = min(max(self.params["open_grasp_width"], action.grasp_width * 3.0), self.params["max_open_grasp_width"])
            # step 1: move to start pose
            pose = manip_traj[0]
            pose = mat44_to_tf6d(pose)
            pose = np.concatenate((pose, [open_grasp_width]))
            wps += self.transit_between_poses(cur_pose, pose, transit_plane_x, self.params["transit_stay"], open_grasp_width)
            key_wps += [False] * len(wps)
            key_wps[0], key_wps[-1] = True, True  # start and end are key waypoints
            # step 2: move to grasp pose
            for t in range(self.params["pre_num_waypoints"]):
                pose = manip_traj[t]
                pose = mat44_to_tf6d(pose)
                pose = np.concatenate((pose, [open_grasp_width]))
                wps.append(pose)
                key_wps.append(False)
            key_wps[-1] = True  # key_wp: pre-grasp pose
            # step 2.5: stay at grasp pose
            for t in range(self.params["grasp_pose_stay"] // 2):
                cur_grasp_width = np.linspace(open_grasp_width, grasp_width, self.params["grasp_pose_stay"] // 2)[t]
                # cur_grasp_width = open_grasp_width
                pose = manip_traj[self.params["pre_num_waypoints"]]
                pose = mat44_to_tf6d(pose)
                pose = np.concatenate((pose, [cur_grasp_width]))
                wps.append(pose)
                key_wps.append(False)
            key_wps[-1] = True  # key_wp: grasp pose
            # step: 3: grasp
            for t in range(self.params["grasp_pose_stay"] // 2):
                pose = manip_traj[self.params["pre_num_waypoints"]]
                pose = mat44_to_tf6d(pose)
                pose = np.concatenate((pose, [grasp_width]))
                wps.append(pose)
                key_wps.append(False)
            # step 4: drag
            for t in range(joint_waypoints):
                pose = manip_traj[t + self.params["pre_num_waypoints"]]
                pose = mat44_to_tf6d(pose)
                pose = np.concatenate((pose, [grasp_width]))
                wps.append(pose)
                key_wps.append(False)
            key_wps[-1] = True  # key_wp: finish draging
            # step 4.5: stay at release pose
            for t in range(self.params["joint_place_stay"]):
                pose = manip_traj[self.params["pre_num_waypoints"] + joint_waypoints]
                pose = mat44_to_tf6d(pose)
                pose = np.concatenate((pose, [grasp_width]))
                wps.append(pose)
                key_wps.append(False)
            # step 5: release
            pose = manip_traj[self.params["pre_num_waypoints"] + joint_waypoints]
            pose = mat44_to_tf6d(pose)
            pose = np.concatenate((pose, [open_grasp_width]))
            wps.append(pose)
            key_wps.append(False)  # key_wp: release pose
            # step 6: draw hand back
            for t in range(self.params["pre_num_waypoints"]):
                pose = manip_traj[t + self.params["pre_num_waypoints"] + joint_waypoints]
                pose = mat44_to_tf6d(pose)
                pose = np.concatenate((pose, [open_grasp_width]))
                wps.append(pose)
                key_wps.append(False)
        elif action.action_type == "rigid":
            goal_space_type = action.goal_space_type
            rigid_obj_name = action.manip_obj
            rigid_idx = self.sampler.get_rigid_idx(rigid_obj_name)
            grasp_pose_local = action.grasp_pose
            grasp_width = action.grasp_width / self.params["grasp_tight_ratio"]
            grasp_pose = self.sampler.rigid_list[rigid_idx].tf @ grasp_pose_local
            # Compute goal pose
            goal_pose = action.goal_pose
            self.sampler.rigid_list[rigid_idx].update_pose(goal_pose[:6], is_raw=True)
            place_pose = self.sampler.rigid_list[rigid_idx].tf @ grasp_pose_local
            place_pose[:3, 3] += self.params["place_lift_offset"] * np.array([0, 0, 1])
            open_grasp_width = min(max(self.params["open_grasp_width"], action.grasp_width * 3.0), self.params["max_open_grasp_width"])
            # print(f"Grasp width: {grasp_width}, Open width: {open_grasp_width}")
            pre_grasp_pose = deepcopy(grasp_pose)
            pre_grasp_pose[:3, 3] -= self.params["pre_offset_grasp"] * grasp_pose[:3, 2]
            pre_grasp_pose = mat44_to_tf6d(pre_grasp_pose)
            pre_grasp_pose = np.concatenate((pre_grasp_pose, [open_grasp_width]))

            # Add all waypoints
            # step 1: transit to pre-grasp pose
            wps += self.transit_between_poses(cur_pose, pre_grasp_pose, transit_plane_x, self.params["transit_stay"], open_grasp_width)
            key_wps += [False] * len(wps)
            key_wps[0], key_wps[-1] = True, True  # start and end are key waypoints
            # step 2: move to grasp pose
            grasp_pose = mat44_to_tf6d(grasp_pose)
            grasp_pose = np.concatenate((grasp_pose, [open_grasp_width]))
            for t in range(self.params["grasp_pose_stay"] // 2):
                # cur_grasp_width = np.linspace(open_grasp_width, grasp_width, self.params["grasp_pose_stay"] // 2)[t]
                cur_grasp_width = open_grasp_width
                grasp_pose_close = deepcopy(grasp_pose)
                grasp_pose_close[6] = cur_grasp_width
                wps.append(grasp_pose_close)
                key_wps.append(False)
            key_wps[-1] = True  # key_wp: grasp pose
            # step 3: grasp
            for t in range(self.params["grasp_pose_stay"] // 2):
                grasp_pose_close = deepcopy(grasp_pose)
                grasp_pose_close[6] = grasp_width
                wps.append(grasp_pose_close)
                key_wps.append(False)
            # step 4: lift to pre-grasp pose
            pre_grasp_pose_close = deepcopy(pre_grasp_pose)
            pre_grasp_pose_close[6] = grasp_width
            wps.append(pre_grasp_pose_close)
            key_wps.append(False)
            # step 5: transit between two poses, move to place pose
            pre_place_pose_close = np.copy(place_pose)
            pre_place_pose_close[:3, 3] -= self.params["pre_offset_place"] * pre_place_pose_close[:3, 2]
            pre_place_pose_close = mat44_to_tf6d(pre_place_pose_close)
            pre_place_pose_close = np.concatenate((pre_place_pose_close, [grasp_width]))
            # compute transit
            z_transit = goal_space_type == "drawer" or goal_space_type == "all"
            _trans_wps = self.transit_between_poses(
                pre_grasp_pose_close,
                pre_place_pose_close,
                transit_plane_x,
                self.params["transit_stay"],
                grasp_width,
                z_transit=z_transit,
            )
            wps += _trans_wps
            key_wps += [False] * len(_trans_wps)
            key_wps[0], key_wps[-1] = True, True  # start and end are key waypoints
            place_pose_close = mat44_to_tf6d(place_pose)
            place_pose_close = np.concatenate((place_pose_close, [grasp_width]))
            wps.append(place_pose_close)
            key_wps.append(True)  # key_wp: place pose
            # step 6: release
            place_pose_open = mat44_to_tf6d(place_pose)
            place_pose_open = np.concatenate((place_pose_open, [open_grasp_width]))
            wps.append(place_pose_open)
            key_wps.append(False)
            # step 7: lift to pre-place pose
            pre_place_pose = deepcopy(place_pose)
            pre_place_pose[:3, 3] -= self.params["pre_offset_place"] * pre_place_pose[:3, 2]
            pre_place_pose_open = mat44_to_tf6d(pre_place_pose)
            pre_place_pose_open = np.concatenate((pre_place_pose_open, [open_grasp_width]))
            wps.append(pre_place_pose_open)
            key_wps.append(True)  # key_wp: pre-place pose; lift up after place
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
        assert len(wps) == len(key_wps), f"Key wps size not match: {len(wps)} != {len(key_wps)}"
        # Interpolate between waypoints
        trajectory = []
        trajectory_kps = []
        for i in range(len(wps) - 1):
            cur_pose = wps[i]
            goal_pose = wps[i + 1]
            inter_traj = self._interpolate(cur_pose, goal_pose)
            if len(inter_traj) > 1:
                trajectory.append(inter_traj[:-1])
                inter_traj_kps = [False] * (len(inter_traj) - 1)
                inter_traj_kps[0] = key_wps[i]
                trajectory_kps += inter_traj_kps
            elif len(inter_traj) == 1:
                trajectory.append(inter_traj)
                trajectory_kps.append(key_wps[i])
            else:
                raise ValueError("Invalid trajectory length.")
        trajectory.append(wps[-1].reshape(1, -1))
        trajectory_kps.append(key_wps[-1])
        trajectory = np.vstack(trajectory)
        assert len(trajectory) == len(trajectory_kps), f"Key wps size not match: {len(trajectory)} != {len(trajectory_kps)}"
        assert sum(trajectory_kps) == sum(key_wps), f"Key waypoints not match: {sum(trajectory_kps)} != {sum(key_wps)}"
        return trajectory, trajectory_kps

    def transit_between_poses(self, start_pose, end_pose, transit_plane_x, transit_stay, grasp_width=0.03, z_transit=False):
        """pose is 6d+grasp"""
        waypoints = []
        waypoints.append(np.copy(start_pose))
        if not z_transit:
            # transit in y-z plane
            # move to transit plane
            transit_plane_x = max(max(start_pose[0], end_pose[0]) + self.params["move_front_offset"], transit_plane_x)
            transit_pose = np.copy(start_pose)
            transit_pose[0] = transit_plane_x
            for t in range(transit_stay // 2):
                waypoints.append(transit_pose)
            # move to end pose in transit plane
            end_pose_transit = np.copy(end_pose)
            end_pose_transit[0] = transit_plane_x
            for t in range(transit_stay // 2):
                waypoints.append(end_pose_transit)
            # move to end pose
            waypoints.append(np.copy(end_pose))
        else:
            # transit in x-y plane
            transit_z = max(start_pose[2], end_pose[2])
            # move to transit plane
            transit_pose = np.copy(start_pose)
            transit_pose[2] = transit_z
            for t in range(transit_stay // 2):
                waypoints.append(transit_pose)
            # move to end pose in transit plane
            end_pose_transit = np.copy(end_pose)
            end_pose_transit[2] = transit_z
            for t in range(transit_stay // 2):
                waypoints.append(end_pose_transit)
            # move to end pose
            waypoints.append(np.copy(end_pose))
        # set grasp width
        for waypoint in waypoints:
            waypoint[6] = grasp_width
        return waypoints
