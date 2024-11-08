"""Sample Interface for LGMCTS
If an action can be executed decided on two things:
1. If this action will not cause collision.
2. If this action can be applied at a reachable space.

For example, open a box, we need to check if open the box will cause collision with other objects.
Then we need to make sure trajectory of openning the box is reachable.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from lgmcts.object_primitives import (
    ArtObject,
    Link,
    Robot,
    RigidObject,
    Space,
    check_collision_SAT,
)
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce

################################## Utils ##################################


class SampleStatus(Enum):
    """Sample status"""

    SUCCESS = 0  # success
    REGION_SMALL = 1  # region is too small
    IN_COLLISION = 2  # collision
    SPACE_NO_REACHABLE = 3  # Space not reachable
    OBJ_NO_REACHABLE = 4  # Object not reachable
    NO_VALID_POSE = 5  # No valid pose
    UNKNOWN = 6  # unknown, placeholder


@dataclass
class SceneState:
    """Sample data structure."""

    rigit_poses: list[np.ndarray] = field(default_factory=list)  # (3D pose + 3D rotate)
    art_poses: list[np.ndarray] = field(default_factory=list)  # (3D pose + 3D rotate + Joint values)


@dataclass
class Action:
    """Action for MCTS. Grasp one object from a pose to another pose."""

    manip_obj: str = None
    manip_joint: str = None
    goal_obj: str = None
    goal_pose: np.ndarray = None
    goal_space: str = None
    goal_space_type: str = None
    # Need to get from motion planning
    grasp_pose: np.ndarray = None
    grasp_width: float = None
    # Different types of actions:
    # art: move articulation object
    # rigid: move rigid object
    action_type: str = "art"
    # Replay related
    replay_file: str = ""


@dataclass
class Goal:
    """Goal for MCTS"""

    manip_obj: str = None
    manip_joint: str = None
    goal_pose: np.ndarray = None
    goal_obj: str = None
    goal_space: str = None
    relation: str = None
    priority: int = 0


class Sampler:

    def __init__(self, joint_itp_steps: int = 5, enable_stick=True, seed=0):
        self.robot: Robot = None
        self.rigid_list: list[RigidObject] = []
        self.art_list: list[ArtObject] = []
        self.spaces: dict[str, Space] = {}
        self.external_spaces: dict[str, Space] = {}
        self.rigid_name2idx: dict[str, int] = {}
        self.art_name2idx: dict[str, int] = {}
        # Require a tree structure
        self.joint_itp_steps = joint_itp_steps
        self.enable_stick = enable_stick  # If rigid object sticks to space
        # Random seed
        self.rng = np.random.RandomState(seed)

    def init_env(
        self,
        robot: Robot = None,
        rigid_list: list[RigidObject] = [],
        art_list: list[ArtObject] = [],
        external_spaces: dict[str, Space] = {},
    ):
        self.robot = robot
        self.art_list = art_list
        # Log the space
        for art_obj in self.art_list:
            self.spaces.update(art_obj.spaces)
            art_obj.forward_pose()
        self.rigid_list = rigid_list
        # Binding rigid object to space
        for idx, rigid_obj in enumerate(self.rigid_list):
            self.set_rigid_obj_pose(idx, rigid_obj.get_pose(), False)
        # Binding space
        for name, space in external_spaces.items():
            self.spaces[name] = space
            self.external_spaces[name] = space
        # Update name2idx
        self.rigid_name2idx = {rigid_obj.name: idx for idx, rigid_obj in enumerate(self.rigid_list)}
        assert len(self.rigid_name2idx) == len(self.rigid_list), "Duplicate rigid name"
        self.art_name2idx = {art_obj.obj_name: idx for idx, art_obj in enumerate(self.art_list)}
        assert len(self.art_name2idx) == len(self.art_list), "Duplicate art name"
        # Compute the visiblity & reachablity
        self.update_status()
        # Return the current state
        return self.get_state()

    def sync_from_state(self, scene_state, scalar_first=True):
        """Sync from isaac-like state.
        For isaacgym, scalar_first is False, for isaaclab, scalar_first is True.
        """
        # Align the lgmcts & issac gym joint order.
        for obj_name in scene_state:
            if obj_name in self.art_name2idx:
                obj_state = scene_state[obj_name]
                obj_pose = np.eye(4)
                obj_pose[:3, 3] = obj_state["state"][:3]
                obj_pose[:3, :3] = R.from_quat(obj_state["state"][3:7], scalar_first=scalar_first).as_matrix()
                art_idx = self.art_name2idx[obj_name]
                # Mapping between two joint values
                isaac_joint_values = obj_state["joint_values"]
                lgmcts_joint_values = np.zeros(len(self.art_list[art_idx].active_joints))
                isaac2lgmcts_idx = self.art_list[art_idx].get_isaac2lgmcts_idx()
                for isaac_idx, lgmcts_idx in isaac2lgmcts_idx.items():
                    lgmcts_joint_values[lgmcts_idx] = isaac_joint_values[isaac_idx]
                self.art_list[art_idx].update_pose(obj_pose, move_center=False)
                # Absolute joint values
                self.set_joint_values(art_idx, lgmcts_joint_values, is_rel=False)
        # Update rigid-obj later
        for obj_name in scene_state:
            if obj_name in self.rigid_name2idx:
                obj_state = scene_state[obj_name]
                obj_pose = np.eye(4)
                obj_pose[:3, 3] = obj_state["state"][:3] + self.rigid_list[self.rigid_name2idx[obj_name]].offset
                obj_pose[:3, :3] = R.from_quat(obj_state["state"][3:7], scalar_first=scalar_first).as_matrix()
                self.set_rigid_obj_pose(self.rigid_name2idx[obj_name], obj_pose)
        self.update_status()

    ##################### Interface #####################
    def get_art_idx(self, name, relax=False):
        """Get the idx of the art obj given name.
        relax: use in for loose match.
        """
        for idx, art_obj in enumerate(self.art_list):
            if relax:
                if name in art_obj.obj_name:
                    return idx
            else:
                if name == art_obj.obj_name:
                    return idx

    def get_art_idx_by_space(self, name, relax=False):
        """Get the idx of the art obj given space name.
        relax: use in for loose match.
        """
        for idx, art_obj in enumerate(self.art_list):
            for space_name in art_obj.spaces:
                if relax:
                    if name in space_name:
                        return idx
                else:
                    if name == space_name:
                        return idx

    def get_rigid_idx(self, name, relax=False):
        """Get the idx of the rigid obj given name.
        relax: use in for loose match.
        """
        for idx, rigid_obj in enumerate(self.rigid_list):
            if relax:
                if name in rigid_obj.name:
                    return idx
            else:
                if name == rigid_obj.name:
                    return idx

    def get_space(self, name, relax=False):
        """Get the space given name.
        relax: use in for loose match.
        """
        space_list = []
        for space_name, space in self.spaces.items():
            if relax:
                if name in space_name:
                    space_list.append(space)
            else:
                if name == space_name:
                    return space
        return space_list

    def set_joint_values(self, idx, joint_values, update_status=True, is_rel=True):
        self.art_list[idx].set_joint_values(joint_values, is_rel=is_rel)
        if update_status:
            self.update_status()

    def set_rigid_obj_pose(self, idx, goal_pose, update_status=True):
        # Interface for set rigid object, automatically handle obj stick
        rigid_obj = self.rigid_list[idx]
        if not self.enable_stick:
            rigid_obj.update_pose(goal_pose)
        else:
            # First detect if it is already sticked
            cur_pose = rigid_obj.get_pose()
            bboxes = self.get_space_belong(cur_pose)
            if len(bboxes) != 0:
                # It is already in a space
                if rigid_obj in self.spaces[bboxes[0]].contains:
                    self.spaces[bboxes[0]].contains.remove(rigid_obj)
            # Attach object to new space
            rigid_obj.update_pose(goal_pose)
            bboxes = self.get_space_belong(goal_pose)
            if len(bboxes) != 0:
                self.spaces[bboxes[0]].contains.append(rigid_obj)
                # Update obj's tf2parent
                rigid_obj.tf2parent = np.linalg.inv(self.spaces[bboxes[0]].tf) @ rigid_obj.tf
                rigid_obj.tf2parent[:3, 3] += rigid_obj.offset
        if update_status:
            self.update_status()

    def set_state(self, state: SceneState, update_status=True):
        for idx, art_pose in enumerate(state.art_poses):
            art_root_pose = art_pose[:6]
            art_joint_pose = art_pose[6:]
            # FIXME: currently when set state, use base pose.
            self.art_list[idx].update_pose(art_root_pose, move_center=False)
            self.art_list[idx].set_joint_values(art_joint_pose)
        for idx, rigid_pose in enumerate(state.rigit_poses):
            # self.rigid_list[idx].update_pose(rigid_pose)
            self.set_rigid_obj_pose(idx, rigid_pose, False)  # Binding interface
        if update_status:
            self.update_status()

    def get_state(self, is_rel=True):
        # Return the current state
        rigid_poses = [rigid_obj.get_pose() for rigid_obj in self.rigid_list]
        art_poses = [np.concatenate([art_obj.get_pose(), art_obj.get_joint_values(is_rel=is_rel)]) for art_obj in self.art_list]
        return SceneState(rigit_poses=rigid_poses, art_poses=art_poses)

    def get_visible_objs(self):
        return list(filter(lambda x: x.is_visible, self.rigid_list))

    def get_reachable_objs(self):
        return list(filter(lambda x: x.is_reachable, self.rigid_list))

    def show(self, ee_traj=[], show_origin=False, show_floor=False):
        """Visualize the current state"""
        import open3d as o3d

        vis = []
        if self.robot is not None:
            vis += self.robot.get_vis_o3d()
        for rigid_obj in self.rigid_list:
            vis += rigid_obj.get_vis_o3d()
        for art_obj in self.art_list:
            vis += art_obj.get_vis_o3d()
        for space in self.spaces.values():
            vis += space.get_vis_o3d()
        # Add trajectory
        for traj in ee_traj:
            ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            ee.transform(traj)
            vis.append(ee)
        if show_origin:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis.append(origin)
        if show_floor:
            floor = o3d.geometry.TriangleMesh.create_box(2, 2, 0.002)
            floor.translate([-1, -1, -0.002])
            floor.compute_vertex_normals()
            floor.paint_uniform_color([0.7, 0.7, 0.7])
            vis.append(floor)
        o3d.visualization.draw_geometries(vis)

    ######################## Space-related function  ########################
    def get_space_belong(self, pose):
        """Check the space belong of a give pos"""
        if pose.shape == (3,):
            pos = pose
        elif pose.shape == (4, 4):
            pos = pose[:3, 3]
        elif pose.shape == (6,):
            pos = pose[:3]
        belong_spaces = []
        for name, space in self.spaces.items():
            if space.check_in(pos):
                bbox_size = np.prod(space.extent)
                belong_spaces.append((bbox_size, name))
        # Sort by bbox size (ascending order)
        belong_spaces.sort(key=lambda x: x[0])
        return [name for _, name in belong_spaces]

    def update_status(self):
        self.update_rigid_obj_status()

    def update_rigid_obj_status(self):
        # Call this when neccessary
        # Internally update states
        for rigid_obj in self.rigid_list:
            spaces = self.get_space_belong(rigid_obj.get_pose())
            cur_visible = rigid_obj.is_visible
            cur_reachable = True
            # update states
            for space in spaces:
                space_visible = self.spaces[space].is_visible
                space_reachable = self.spaces[space].is_reachable
                cur_reachable = cur_reachable and space_reachable
                cur_visible = cur_visible or space_visible
            rigid_obj.is_reachable = cur_reachable
            rigid_obj.is_visible = cur_visible

    ################################## Sampling ##################################
    def sample_poses(
        self,
        obj_idx,
        space_name="",
        n_samples=1,
        rng=np.random.default_rng(),  # rng=np.random.RandomState(0) NOTE: temp change here made
        **kwargs,
    ):
        """Sample a pose for one object to a given space or all spaces.
        Return a list of 4x4 poses.
        """
        # To be implemented
        obj_extent = self.rigid_list[obj_idx].extent
        # Sample from every spaces
        sample_poses = []
        for _name, space in self.spaces.items():  # make random
            if space_name in _name:
                _sample_poses = space.sample_poses(
                    obj_extent,
                    n_samples=n_samples,
                    rng=rng,
                    **kwargs,
                )
                if _sample_poses is not None:
                    sample_poses.append(_sample_poses)
        if len(sample_poses) == 0:
            return None
        else:
            return np.concatenate(sample_poses, axis=0)

    ################################## Feasibility Check ##################################
    def check_feasible(
        self,
        obj_idx,
        goal_pose,
        cur_state: SceneState = None,
        is_art=False,
        is_store_act=False,
    ):
        """
        Check if one action is feasible.
        Args:
            obj_idx: object index.
            goal_pose: object pose. This is majorly used for checking collision.
            is_art: if the object is an articulation object.
            is_store_act: if this action is a storage action. This needs to check if the corresponding storage
               is connected to the outer space.
            fix_state: wheather fix the state during the check_feasible process
        """
        assert not (is_art and is_store_act), "Articulation object cannot be stored."
        # Check feasible of the space
        spaces_not_reach = []
        if not is_art:
            # For rigid object, the goal space must be reachable
            belong_spaces = self.get_space_belong(goal_pose)
            for space_name in belong_spaces:
                if not self.spaces[space_name].is_reachable:
                    spaces_not_reach.append(space_name)
        # FIXME: How to handle case of store an articulated object needs to be investigate.
        if not len(spaces_not_reach) == 0:
            return SampleStatus.SPACE_NO_REACHABLE, spaces_not_reach, []

        collision_status, collision_obj, collision_art_obj = self.check_collision(obj_idx, goal_pose, cur_state, is_art)
        if collision_status != SampleStatus.SUCCESS:
            # Recover the state
            return collision_status, collision_obj, collision_art_obj
        else:
            return SampleStatus.SUCCESS, [], []

    def check_collision(self, obj_idx, goal_pose, cur_state=None, is_art=False, ignore_storage=True):
        """
        Check if one action is feasible.
        Args:
            obj_idx: object index.
            goal_pose: object pose & joint values. This is majorly used for checking collision.
        """
        if cur_state is not None:
            self.set_state(cur_state)
        _cur_state = self.get_state()  # Record current state
        if not is_art:
            # Object is rigid
            rigid_bboxes = [rigid_obj.get_bbox() for rigid_obj in self.rigid_list]
            art_bboxes = sum([art_obj.get_bboxes() for art_obj in self.art_list], [])
            art_storage_properties = sum([art_obj.get_storage_properties() for art_obj in self.art_list], [])
            # Check collision with rigid objects
            self.set_rigid_obj_pose(obj_idx, goal_pose)
            goal_bbox = self.rigid_list[obj_idx].get_bbox()
            rigid_collision_status = check_collision_SAT(goal_bbox, rigid_bboxes)
            rigid_collision_status[obj_idx] = False  # Ignore self-collision
            art_collision_status = check_collision_SAT(goal_bbox, art_bboxes)
            if ignore_storage:
                art_collision_status = [art_collision_status[i] and not art_storage_properties[i] for i in range(len(art_collision_status))]
            if np.any(rigid_collision_status) or np.any(art_collision_status):
                # NOTE: return which objects colliding ####
                collision_objects = []
                for i in range(len(rigid_collision_status)):
                    if rigid_collision_status[i] == True:
                        collision_objects.append(i)
                art_collision_objects = []
                for i in range(len(art_collision_status)):
                    if art_collision_status[i] == True:
                        art_collision_objects.append(i)
                #######
                self.set_state(_cur_state)  # Recover init state
                return (
                    SampleStatus.IN_COLLISION,
                    collision_objects,
                    art_collision_objects,
                )
            else:
                self.set_state(_cur_state)  # Recover init state
                return SampleStatus.SUCCESS, [], []
        else:
            # We only test the collision between articulation & rigid objects
            # Not the collision between articulation objects
            # Ignore contained objects
            cur_joint_values = self.art_list[obj_idx].get_joint_values()
            goal_joint_values = goal_pose[6:]
            contained_objs = self.art_list[obj_idx].get_contained_objs()
            checking_objs = [obj for obj in self.rigid_list if obj not in contained_objs]
            if len(checking_objs) == 0:
                self.set_state(_cur_state)  # Recover init state
                return SampleStatus.SUCCESS, [], []
            # Check collision for itp steps
            for idx in range(self.joint_itp_steps):
                joint_values = (cur_joint_values * (self.joint_itp_steps - idx) + goal_joint_values * idx) / self.joint_itp_steps
                self.art_list[obj_idx].set_joint_values(joint_values)
                art_bboxes = sum([self.art_list[obj_idx].get_bboxes()], [])
                rigid_bboxes = [rigid_obj.get_bbox() for rigid_obj in checking_objs]
                art_collision_status = []
                # FIXME: this can be a mismatch
                for art_idx, art_bbox in enumerate(art_bboxes):
                    # if self.art_list[obj_idx].check_storage(art_idx) and ignore_storage:
                    #     # Skip the storage part; As it is considered as in collision
                    #     continue
                    art_collision_status = check_collision_SAT(art_bbox, rigid_bboxes)
                    if np.any(art_collision_status):
                        # NOTE: return which objects colliding ####
                        collision_objects = []
                        # print(art_collision_status)
                        for i in range(len(art_collision_status)):
                            if art_collision_status[i] == True:
                                collision_objects.append(i)
                        #######
                        self.set_state(_cur_state)  # Recover init state
                        return SampleStatus.IN_COLLISION, collision_objects, []
            self.set_state(_cur_state)  # Recover init state
            return SampleStatus.SUCCESS, [], []

    def check_storage(self, obj_idx, goal_pose, cur_state):
        """
        Check if one action is feasible. Storage can be only appled to non-articulation object.
        Args:
            obj_idx: object index.
            goal_pose: object pose. This is majorly used for checking collision.
        """
        # To be implemented
        return SampleStatus.SUCCESS


######################## Eval related  ########################
def check_goal(sampler: Sampler, _goal: Goal | dict):
    if isinstance(_goal, dict):
        goal = Goal(**_goal)
    else:
        goal = _goal
    # Check if the goal is satisfied
    if goal.relation == "in":
        manip_obj = goal.manip_obj
        goal_space = goal.goal_space
        rigid_obj_pose = sampler.rigid_list[sampler.get_rigid_idx(manip_obj)].get_pose(is_raw=False)
        space_belong = sampler.get_space_belong(rigid_obj_pose)
        # Check the overlap of goal space
        if any([goal_space in space for space in space_belong]):
            return True
    elif goal.relation == "open":
        manip_obj = goal.manip_obj
        manip_joint = goal.manip_joint
        manip_joint_idx = sampler.art_list[sampler.get_art_idx(manip_obj, relax=True)].active_joints.index(manip_joint)
        joint_pos = sampler.art_list[sampler.get_art_idx(manip_obj, relax=True)].get_joint_values(is_rel=True)[manip_joint_idx]
        if joint_pos > 0.9:
            return True
    return False
