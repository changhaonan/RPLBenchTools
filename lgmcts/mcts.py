from __future__ import annotations
from typing import List, Any, Dict
import numpy as np
import math
import copy
import open3d as o3d
import os
import yaml
import sys
from lgmcts.sampler import Action, Goal, Sampler, SceneState, SampleStatus, check_goal
from scipy.spatial.transform import Rotation as R
from lgmcts.tasks import load_task_from_config_file
from lgmcts.motion_planner import find_path_pick_place, find_path_manip_joint
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from lgmcts.object_primitives import (
    ArtObject,
    Link,
    RigidObject,
    Space,
    load_art_object_from_config,
    load_link_from_config,
    load_space_from_config,
)


######################################## Utils functions ########################################
def mat2vec(matrix_4x4):
    rotation_matrix = matrix_4x4[:3, :3]
    translation_vector = matrix_4x4[:3, 3]
    rotation = R.from_matrix(rotation_matrix)
    rotation_vector = rotation.as_rotvec()
    return np.concatenate((translation_vector, rotation_vector))


def visualize_mcts(mcts):
    # Directed graph
    G = nx.DiGraph()
    queue = [(mcts.root, None, None)]  # (current_node, parent_node, action)
    node_depths = {mcts.root.node_id: 0}

    while queue:
        current_node, parent_node, action = queue.pop(0)
        if current_node.node_id not in node_depths:
            if parent_node is not None:
                node_depths[current_node.node_id] = node_depths[parent_node.node_id] + 1
            else:
                node_depths[current_node.node_id] = 0
        # Generate label
        node_label = f"Node {current_node.node_id}"  # ID
        node_label += f"\nReward: {current_node.total_reward}"  # Reward
        node_action = current_node.action_from_parent
        if node_action is not None:
            node_label += f"\nAction: {node_action.action_type}"
            node_label += f"\nManip obj: {node_action.manip_obj}"
            node_label += f"\nGoal obj: {node_action.goal_obj}"
            node_label += f"\nGoal space: {node_action.goal_space}"
            if node_action.action_type == "art":
                node_label += f"\nManip joint: {node_action.manip_joint}"
        G.add_node(
            current_node.node_id,
            label=node_label,
            subset=node_depths[current_node.node_id],
        )
        if parent_node is not None:
            G.add_edge(parent_node.node_id, current_node.node_id, label=str(action))
        for action_key, child_node in current_node.children.items():
            if child_node.node_id not in node_depths:
                node_depths[child_node.node_id] = node_depths[current_node.node_id] + 1
            queue.append((child_node, current_node, action_key))

    # Draw graph
    pos = nx.multipartite_layout(G, subset_key="subset")
    nx.draw(G, pos, with_labels=True, labels={n: G.nodes[n]["label"] for n in G.nodes}, node_size=2000, node_color="lightblue", font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G.edges[u, v]["label"] for u, v in G.edges}, font_size=8)
    plt.show()


@dataclass
class NodeData:
    node_id: int
    action: Action
    state: SceneState
    goal_targets: list[Goal]
    completed_goals: list[Goal]
    status: SampleStatus = SampleStatus.SUCCESS


######################################## MCTS classes ########################################
class Node:
    """MCTS Node"""

    _id_counter = 0

    @staticmethod
    def get_next_id():
        Node._id_counter += 1
        return Node._id_counter

    def __init__(
        self,
        node_id: int = None,
        parent: "Node" = None,
        original_state: SceneState = None,
        action_from_parent: Any = Action(),
        UCB_scalar: float = 1.0,
        num_sampling: int = 1,
        reward_mode: str = "same",
        rng: np.random.Generator = None,
        goal_config: Dict = None,
        goal_targets=[],
        completed_goals=[],
        sampler: Sampler = None,
        action_objects=[],
        action_positions=[],
        verbose: bool = False,
        mp_config: dict = {},
    ) -> None:
        self.parent = parent
        self.node_id = Node.get_next_id()
        self.state = original_state  # inherited from final state of parent
        self.action_from_parent = action_from_parent
        self.UCB_scalar = UCB_scalar
        self.num_sampling = num_sampling
        self.reward_mode = reward_mode
        self.rng = rng
        self.sampler = copy.deepcopy(sampler)
        self.sampler.set_state(self.state)
        self.goal_config: list[Goal] = goal_config
        self.goal_targets: list[Goal] = goal_targets
        self.completed_goals: list[Goal] = completed_goals

        self.children = {}
        self.unvisited_actions: list[Action] = self.generate_actions() * self.num_sampling
        self.visited_actions = []
        self.visited_time = 0
        self.total_reward = 0
        self.action_objects = action_objects
        self.action_positions = action_positions
        self.action_relax = False
        self.art_only_one_joint = True  # Only allow one joint to be opened at a time

        self.mp_config = mp_config
        # [DEBUG]
        self.verbose = verbose

    def generate_actions(self):
        """
        Generate the list of actions for this node, all articulation structures and specific objects and their goal locations
        """
        unvisited_actions = []
        # add actions of manipulating articulation objects
        for art_obj in self.sampler.art_list:
            for joint_name in art_obj.active_joints:
                action = Action(manip_obj=art_obj.obj_name, manip_joint=joint_name, action_type="art")
                if action.action_type == self.action_from_parent.action_type == "art":
                    if action.manip_obj == self.action_from_parent.manip_obj and action.manip_joint == self.action_from_parent.manip_joint:
                        continue
                # Jump over the same action
                unvisited_actions.append(action)

        # FIXME: what is this???
        cur_priority = math.inf
        for goal in self.goal_targets:
            if goal.priority < cur_priority:
                cur_priority = goal.priority

        # for rigid objects / links, append objects that pertain to goal config ()
        # iterate through target list and for tuples append specific goals
        # see -> goal_config = {"apple": [("base", ["drawer_0"])]}
        for rigid_obj in self.sampler.rigid_list:
            if rigid_obj.name in self.goal_config:
                if rigid_obj in self.sampler.get_reachable_objs():  # if object is reachable it is visible
                    goal = self.goal_config[rigid_obj.name]
                    # check if obj is already in goal state, if it is no need for action
                    if goal.priority != cur_priority:
                        # only deal with the current priority
                        continue
                    if not check_goal(self.sampler, goal):
                        # if the goal is not satisfied, add to the action list
                        #
                        if goal.relation == "in":
                            goal_spaces = self.sampler.get_space(goal.goal_space, relax=True)
                            for goal_space in goal_spaces:
                                action = Action(
                                    manip_obj=rigid_obj.name,
                                    goal_obj=goal.goal_obj,
                                    goal_space=goal_space.name,
                                    goal_space_type=goal_space.space_type,
                                    action_type="rigid",
                                )
                                unvisited_actions.append(action)
                        elif goal.relation == "open":
                            pass
        return unvisited_actions

    def UCB(self):
        """Upper confidence bound"""
        assert len(self.unvisited_actions) == 0
        best_action = None
        best_sample = None
        best_reward = -float("inf")
        for action, child in self.children.items():
            reward = (child.total_reward / child.visited_time) + self.UCB_scalar * math.sqrt(2 * math.log10(self.visited_time) / child.visited_time)

            if reward >= best_reward:
                best_reward = reward
                best_action = action
                best_sample = child
        assert best_sample is not None, "No best sample found"
        return best_sample

    def expansion(self):
        """Expand the MCTS tree"""
        sampler_id = self.rng.choice(len(self.unvisited_actions))
        action = self.unvisited_actions.pop(sampler_id)
        node_data = self.action_parametrization(action)

        return node_data

    def action_parametrization(self, action: Action):
        """Convert high-level action to low-level action"""
        if action.action_type == "rigid":
            node_data, info = self.move_rigid_obj(action, relax=self.action_relax)
            if self.verbose:
                print(f"Action: {action}")
                print(f"Status: {node_data.status}")
            if node_data.status == SampleStatus.SUCCESS:
                # Sucessful execution
                return node_data
            elif node_data.status == SampleStatus.IN_COLLISION:
                if len(info) == 0:
                    # No collision, other issues
                    return node_data
                # In collision
                collision_art, collision_obj, in_art = info["collision_art"], info["collision_obj"], info["in_art"]
                if len(collision_art) == 0:
                    # Only in collision with rigid object
                    # Throw the collision object to an external space
                    buffer_action = copy.deepcopy(action)
                    collision_obj_to_move = collision_obj[0]
                    space_name = self.rng.choice(list(self.sampler.external_spaces.keys()))
                    buffer_action.goal_space = space_name
                    buffer_action.goal_space_type = self.sampler.external_spaces[space_name].space_type
                    buffer_action.manip_obj = self.sampler.rigid_list[collision_obj_to_move].name
                    node_data, info = self.move_rigid_obj(buffer_action, relax=self.action_relax)
                    return node_data
            elif node_data.status == SampleStatus.SPACE_NO_REACHABLE:
                in_art = info["in_art"]
                # Currently already in an object, so move it to an external space first
                if in_art:
                    # sampling into articulation object but in articulation object already
                    # (e.g. moving object from drawer 0 to drawer 1, must move to external space first)
                    buffer_action = copy.deepcopy(action)
                    buffer_action.goal_space = self.rng.choice(list(self.sampler.external_spaces.keys()))
                    buffer_action.goal_space_type = self.sampler.external_spaces[buffer_action.goal_space].space_type
                    node_data, info = self.move_rigid_obj(buffer_action, relax=self.action_relax)
                    return node_data
                else:
                    # Open the space
                    space_not_reach = info["space_not_reach"][0]
                    buffer_action = copy.deepcopy(action)
                    # Try to change the status of the space
                    space = self.sampler.get_space(space_not_reach, relax=False)  # "The" space
                    trigger_link = space.trigger[0][0]  # The first trigger
                    art_idx = self.sampler.get_art_idx_by_space(space.name)
                    art_obj_name = self.sampler.art_list[art_idx].obj_name
                    # Flip the joint
                    joint_action = Action(
                        manip_obj=art_obj_name,
                        manip_joint=trigger_link,
                    )
                    node_data, info = self.move_art_obj(joint_action)
                    return node_data
            elif node_data.status == SampleStatus.NO_VALID_POSE:
                # No valid pose
                return node_data
            else:
                raise NotImplementedError(f"Unknown status: {node_data.status}")
        elif action.action_type == "art":
            node_data, info = self.move_art_obj(action)
            if self.verbose:
                print(f"Action: {action}")
                print(f"Status: {node_data.status}")
            if node_data.status == SampleStatus.SUCCESS:
                # Sucessful execution
                return node_data
            elif node_data.status == SampleStatus.IN_COLLISION:
                # In collision
                collision_art, collision_obj = info["collision_art"], info["collision_obj"]
                if len(collision_art) == 0:
                    # self.sampler.show()
                    # Only in collision with rigid object
                    # Throw the collision object to an external space
                    buffer_action = copy.deepcopy(action)
                    collision_obj_to_move = collision_obj[0]
                    space_name = self.rng.choice(list(self.sampler.external_spaces.keys()))
                    buffer_action.action_type = "rigid"
                    buffer_action.manip_obj = self.sampler.rigid_list[collision_obj_to_move].name
                    buffer_action.manip_joint = None
                    buffer_action.goal_space = space_name
                    buffer_action.goal_space_type = self.sampler.external_spaces[space_name].space_type
                    node_data, info = self.move_rigid_obj(buffer_action, relax=self.action_relax)
                    return node_data
                else:
                    # In collision with articulation object
                    raise ValueError("Collision with rigid object and articulation object.")
            elif node_data.status == SampleStatus.NO_VALID_POSE:
                # No valid pose
                return node_data
            else:
                raise NotImplementedError(f"Unknown status: {node_data.status}")

    def move_rigid_obj(self, action: Action, relax: bool = False, debug=False):
        """Execube an action to move a rigid object to a goal space.
        it will be set as False, when moving away obstacles to prevent loop.
        """
        # Locate obj & space
        rigid_idx = self.sampler.get_rigid_idx(action.manip_obj, relax=relax)
        target_space = self.sampler.get_space(action.goal_space, relax=relax)  # Match a goal space loosely
        # Overwrite the goal space
        action.goal_space = target_space.name
        action.goal_space_type = target_space.space_type
        # Analysis status
        original_pose = mat2vec(self.sampler.rigid_list[rigid_idx].tf)
        original_space = self.sampler.get_space_belong(original_pose)
        # No reachable space
        original_space_nr = [self.sampler.spaces[space_name] for space_name in original_space if not self.sampler.spaces[space_name].is_reachable]
        in_articulated_object = any(any(space in art_object.spaces for space in original_space_nr) for art_object in self.sampler.art_list)
        if not self.sampler.rigid_list[rigid_idx] in self.sampler.get_reachable_objs():
            if self.verbose:
                print("[Warning] Object is not reachable")
            return (
                NodeData(self.node_id, action, self.state, self.goal_targets, self.completed_goals, SampleStatus.OBJ_NO_REACHABLE),
                {"in_art": in_articulated_object},
            )
        # Run motion planner to find path
        valid_grasp_goal_pairs = find_path_pick_place(
            sampler=self.sampler,
            obj_idx=rigid_idx,
            goal_space_name=target_space.name,
            space_type=target_space.space_type,
            max_candidates=1,
            rng=self.rng,
            debug=debug,
            mp_config=self.mp_config,
        )
        if len(valid_grasp_goal_pairs) == 0:
            if self.verbose:
                print("[Warning] No valid grasp-goal pairs found.")
            return (
                NodeData(self.node_id, action, self.state, self.goal_targets, self.completed_goals, SampleStatus.NO_VALID_POSE),
                {"in_art": in_articulated_object},
            )

        # Extract the valid action
        grasp_pose, grasp_width, sampled_pose = valid_grasp_goal_pairs[0]
        sampled_pose_vec = mat2vec(sampled_pose)
        # Check feasibility first
        f_status, collision_obj, collision_art = self.sampler.check_feasible(
            obj_idx=rigid_idx,
            goal_pose=sampled_pose_vec,
            cur_state=copy.deepcopy(self.state),
            is_art=False,
            is_store_act=True,
        )
        # Try to proceed with the first feasible action
        if f_status == SampleStatus.SUCCESS:
            sampled_pose_vec = mat2vec(sampled_pose)
            # Update action
            action.grasp_pose = grasp_pose
            action.grasp_width = grasp_width
            action.goal_pose = sampled_pose_vec
            # Successfull placement into sampled position
            updated_sampler = copy.deepcopy(self.sampler)
            updated_sampler.set_rigid_obj_pose(rigid_idx, sampled_pose)
            updated_state = updated_sampler.get_state()
            # Update goal status
            updated_goal_targets = []
            updated_completed_goals = []
            for goal_idx, goal in enumerate(self.goal_targets):
                if check_goal(updated_sampler, goal):
                    updated_completed_goals.append(goal)
                else:
                    updated_goal_targets.append(goal)
            return NodeData(self.node_id, action, updated_state, updated_goal_targets, updated_completed_goals, f_status), {}
        elif f_status == SampleStatus.IN_COLLISION:
            return (
                NodeData(self.node_id, action, self.state, self.goal_targets, self.completed_goals, f_status),
                {"collision_art": collision_art, "collision_obj": collision_obj, "in_art": in_articulated_object},
            )
        elif f_status == SampleStatus.SPACE_NO_REACHABLE:
            space_not_reach = collision_obj
            return (
                NodeData(self.node_id, action, self.state, self.goal_targets, self.completed_goals, f_status),
                {"space_not_reach": space_not_reach, "in_art": in_articulated_object},
            )
        else:
            raise ValueError("Unknown status.")

    def move_art_obj(self, action: Action, relax: bool = False):
        """Move an articulation object to a goal joint_states."""
        art_idx = self.sampler.get_art_idx(action.manip_obj, relax=relax)
        art_object_pose = mat2vec(self.sampler.art_list[art_idx].tf)
        init_joint_values = self.sampler.art_list[art_idx].get_joint_values()
        goal_joint_values = np.copy(init_joint_values)
        joint_idx = self.sampler.art_list[art_idx].active_joints.index(action.manip_joint)
        if self.art_only_one_joint:
            # Only allow one joint to opened
            goal_joint_values = np.zeros(len(init_joint_values))
        goal_joint_values[joint_idx] = 1 - goal_joint_values[joint_idx]  # Flip the joint

        valid_grasp_goal_pairs = find_path_manip_joint(
            self.sampler, art_idx, joint_idx, goal_joint_values[joint_idx], rng=self.rng, mp_config=self.mp_config
        )
        if len(valid_grasp_goal_pairs) == 0:
            if self.verbose:
                print("[Warning] No valid grasp-goal pairs found.")
            return (
                NodeData(self.node_id, action, self.state, self.goal_targets, self.completed_goals, SampleStatus.NO_VALID_POSE),
                {},
            )
        else:
            # Check feasibility first
            goal_pose = np.concatenate((art_object_pose, goal_joint_values))
            f_status, collision_obj, collision_art = self.sampler.check_feasible(
                obj_idx=art_idx,
                goal_pose=goal_pose,
                cur_state=copy.deepcopy(self.state),
                is_art=True,
                is_store_act=False,
            )
            if f_status == SampleStatus.IN_COLLISION:
                return (
                    NodeData(self.node_id, action, self.state, self.goal_targets, self.completed_goals, f_status),
                    {"collision_art": collision_art, "collision_obj": collision_obj},
                )
            elif f_status == SampleStatus.SUCCESS:
                grasp_pose, grasp_width = valid_grasp_goal_pairs[0]
                # Update action
                action.grasp_pose = grasp_pose
                action.grasp_width = grasp_width
                action.goal_pose = goal_pose
                # Successfull manipulate art joint
                updated_sampler = copy.deepcopy(self.sampler)
                updated_sampler.set_joint_values(art_idx, goal_joint_values)
                updated_state = updated_sampler.get_state()
                # Update goal status
                updated_goal_targets = []
                updated_completed_goals = []
                for goal_idx, goal in enumerate(self.goal_targets):
                    if check_goal(updated_sampler, goal):
                        updated_completed_goals.append(goal)
                    else:
                        updated_goal_targets.append(goal)
                return NodeData(self.node_id, action, updated_state, updated_goal_targets, updated_completed_goals, f_status), {}
                # return NodeData(self.node_id, action, cur_state, self.goal_targets, self.completed_goals, f_status), {}
            else:
                return (
                    NodeData(self.node_id, action, self.state, self.goal_targets, self.completed_goals, f_status),
                    {"collision_art": collision_art, "collision_obj": collision_obj},
                )


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(
        self,
        initial_state: SceneState,
        sampler: Sampler,
        goal_config: list[Goal],
        UCB_scalar: float = 3.0,
        reward_mode: str = "same",  # 'same' or 'prop'
        n_samples: int = 1,
        seed: int = 0,
        completed_goals: list = [],
        verbose: bool = False,
        mp_config: dict = {},
    ) -> None:
        goal_targets, completed_goals = self.evaluate_goal(sampler, goal_config)
        self.rng = np.random.default_rng(seed=seed)
        self.settings = {
            "UCB_scalar": UCB_scalar,
            "num_sampling": n_samples,
            "reward_mode": reward_mode,
            "rng": self.rng,
            "sampler": sampler,
            "goal_config": goal_config,
            "verbose": verbose,
            "mp_config": mp_config,
        }
        self.initial_state = initial_state
        self.targets = []
        self.completed_goals = completed_goals
        self.verbose = verbose
        self.root = Node(
            parent=None,
            original_state=initial_state,
            action_from_parent=Action(),
            goal_targets=goal_targets,
            completed_goals=completed_goals,
            **self.settings,
        )

        # Outputs
        self.action_list = []
        self.is_feasible = False
        self.num_iter = 0

    def reset(self):
        """Reset the MCTS"""
        self.root = Node(
            parent=None,
            state=self.initial_state,
            action_from_parent=Action(),
            **self.settings,
        )
        self.action_list = []
        self.is_feasible = False
        self.num_iter = 0

    def evaluate_goal(self, sampler, goal_config: list[Goal]):
        """remove objects already in goal pose from targets"""
        left_goals = []
        completed_goals = []
        for goal in goal_config.values():
            if check_goal(sampler, goal):
                completed_goals.append(goal)
            else:
                left_goals.append(goal)
        return left_goals, completed_goals

    def search(self, max_iter: int = 1000, log_step: int = 1000, debug: bool = False) -> bool:
        """Search for a feasible plan"""
        num_iter = 0
        vis_per_iter = 1000
        while num_iter < max_iter:
            if (num_iter % log_step) == 0:
                if self.verbose:
                    print(f"Searched {num_iter}/{max_iter} iterations")
            num_iter += 1
            current_node = self.selection()
            node_data = current_node.expansion()
            if node_data.status != SampleStatus.SUCCESS:
                # Dead end
                continue
            new_node = self.move(node_data.action, node_data.state, current_node, node_data.goal_targets, node_data.completed_goals)
            current_node.children[new_node.node_id] = new_node
            # new_node.sampler.show()
            reward = self.reward_detection(new_node)
            self.back_propagation(new_node, reward)
            if len(new_node.goal_targets) == 0:
                self.is_feasible = True
                self.construct_plan(new_node)
                # if debug:
                #     visualize_mcts(self)
                return True
            if num_iter % vis_per_iter == 0:
                if debug:
                    visualize_mcts(self)

        self.construct_best_plan()
        self.num_iter = num_iter
        return False

    def selection(self):
        """Select a node to expand"""
        current_node = self.root
        while len(current_node.unvisited_actions) == 0:
            current_node = current_node.UCB()
        return current_node

    def move(
        self,
        action_key,
        new_state,
        current_node,
        goal_targets,
        completed_goals,
    ):
        """Move to a new state and generate a new node"""
        new_node = Node(
            parent=current_node,
            original_state=new_state,
            action_from_parent=action_key,
            goal_targets=goal_targets,
            completed_goals=completed_goals,
            **self.settings,
        )
        return new_node

    def reward_detection(self, node: Node):
        """Detect reward"""
        goal_reward = len(node.completed_goals)
        # Information reward
        # Some objects are intialized as not visible
        # Collecting information are rewarded
        info_reward = 0.1 * (len(node.sampler.get_visible_objs()) - len(self.root.sampler.get_visible_objs()))
        reward = goal_reward + info_reward
        return reward

    def back_propagation(self, node: Node, reward):
        """Backpropagation in MCTS"""
        current_node = node
        while current_node is not None:
            current_node.visited_time += 1
            current_node.total_reward += reward
            current_node = current_node.parent

    def construct_plan(self, node: Node):
        """Construct the plan from the node to the root"""
        self.action_list = []
        current_node = node

        while current_node.parent is not None:
            self.action_list.append(current_node.action_from_parent)
            current_node = current_node.parent
        self.action_list.reverse()

    def construct_best_plan(self):
        """Get the best plan found so far"""
        best_node = self.root
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if len(node.goal_targets) < len(best_node.goal_targets):
                best_node = node
            queue.extend(node.children.values())
        self.construct_plan(best_node)
