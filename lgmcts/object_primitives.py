"""Object primitives; ArtObject; Including translation and rotation object."""

from __future__ import annotations
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


def calculate_zy_rotation_for_arrow(vec):
    """
    Math utils

    Args:
        - vec ():
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1] / (vec[0] + 1e-6))
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )
    # Rotate vec to calculate next rotation
    vec = Rz.T @ vec.reshape(-1, 1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0] / (vec[2] + 1e-6))
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    return (Rz, Ry)


def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):
    import open3d as o3d

    assert not np.all(end == origin)
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    if np.linalg.norm(vec - np.array([0, 0, -1])) < 1e-6:
        Rz = np.eye(3)
        Ry = R.from_euler("y", np.pi).as_matrix()
    else:
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=size / 17.5 * scale,
        cone_height=size * 0.2 * scale,
        cylinder_radius=size / 30 * scale,
        cylinder_height=size * (1 - 0.2 * scale),
    )
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return mesh


################################## Collision computation ##################################
def get_corners(box_min, box_max, position, rotation_matrix):
    # Create 8 corners of the bounding box in local space
    corners = np.array(
        [
            [box_min[0], box_min[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_max[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_max[0], box_max[1], box_max[2]],
        ]
    )
    # Rotate and translate the corners to world space
    transformed_corners = np.dot(corners, rotation_matrix.T) + position
    return transformed_corners


def get_separating_axes(corners1, corners2):
    edges1 = [
        corners1[1] - corners1[0],
        corners1[2] - corners1[0],
        corners1[4] - corners1[0],
    ]
    edges2 = [
        corners2[1] - corners2[0],
        corners2[2] - corners2[0],
        corners2[4] - corners2[0],
    ]
    # Cross
    cross_axes = [np.cross(edge1, edge2) for edge1 in edges1 for edge2 in edges2]
    self_axes = edges1 + edges2
    return np.array(cross_axes + self_axes)


def project(corners, axis):
    projections = np.dot(corners, axis)
    return np.min(projections), np.max(projections)


def check_bbox_collision(box1_min, box1_max, pos1, orn1, box2_min, box2_max, pos2, orn2):
    corners1 = get_corners(box1_min, box1_max, pos1, orn1)
    corners2 = get_corners(box2_min, box2_max, pos2, orn2)

    axes = get_separating_axes(corners1, corners2)
    for axis in axes:
        if np.linalg.norm(axis) < 1e-6:  # Ignore near-zero length axes
            continue
        axis /= np.linalg.norm(axis)
        min1, max1 = project(corners1, axis)
        min2, max2 = project(corners2, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True


def check_collision_SAT_single(bbox_source, bbox_target):
    """Check collision between two bounding boxes."""
    # Source
    bbox_center_s, bbox_rotvec_s, bbox_extent_s = bbox_source
    bbox_min_s = -bbox_extent_s / 2
    bbox_max_s = bbox_extent_s / 2
    bbox_rot_s = R.from_rotvec(bbox_rotvec_s).as_matrix()
    # Target
    bbox_center_t, bbox_rotvec_t, bbox_extent_t = bbox_target
    bbox_min_t = -bbox_extent_t / 2
    bbox_max_t = bbox_extent_t / 2
    bbox_rot_t = R.from_rotvec(bbox_rotvec_t).as_matrix()
    # Check collision
    return check_bbox_collision(
        bbox_min_s,
        bbox_max_s,
        bbox_center_s,
        bbox_rot_s,
        bbox_min_t,
        bbox_max_t,
        bbox_center_t,
        bbox_rot_t,
    )


def check_collision_SAT(bbox_source, bboxes_target):
    """Check collision using Separating Axis Theorem.
    bbox_source is a bbox and bboxes_target is a list of bboxes.
    """
    collision_statuses = []
    for bbox_target in bboxes_target:
        if not check_collision_SAT_single(deepcopy(bbox_source), deepcopy(bbox_target)):
            collision_statuses.append(False)
        else:
            collision_statuses.append(True)
    return collision_statuses


def get_non_overlap_bbox_single(bbox_source, bbox_target, proj_axis: str = "z"):
    """We have a source bbox and a target bbox.
    FIXME: Currently, we use a very simple but coarse method to solve.
    proj_axis: the axis that we project the bbox to. "x", "y", "z", "none".
    """
    assert proj_axis == "y", "Only support y axis now."
    collision = check_collision_SAT_single(bbox_source, bbox_target)
    if not collision:
        return bbox_source
    else:
        # Source
        bbox_center_s, bbox_rotvec_s, bbox_extent_s = bbox_source
        bbox_min_s = -bbox_extent_s / 2
        bbox_max_s = bbox_extent_s / 2
        bbox_rot_s = R.from_rotvec(bbox_rotvec_s).as_matrix()
        # Target
        bbox_center_t, bbox_rotvec_t, bbox_extent_t = bbox_target
        bbox_min_t = -bbox_extent_t / 2
        bbox_max_t = bbox_extent_t / 2
        bbox_rot_t = R.from_rotvec(bbox_rotvec_t).as_matrix()
        # Corners
        corners_s = get_corners(bbox_min_s, bbox_max_s, bbox_center_s, bbox_rot_s)
        corners_t = get_corners(bbox_min_t, bbox_max_t, bbox_center_t, bbox_rot_t)
        # corners_t in s frame
        corners_t_s = bbox_rot_s.T @ (corners_t - bbox_center_s).T
        corners_t_s_max = np.max(corners_t_s, axis=1)
        corners_t_s_min = np.min(corners_t_s, axis=1)
        # Compute the non-overlap part for each axis
        region_nno = np.zeros((3, 2))
        for i in range(3):
            if corners_t_s_max[i] > bbox_max_s[i]:
                if corners_t_s_min[i] > bbox_min_s[i]:
                    region_nno[i] = [bbox_min_s[i], corners_t_s_min[i]]
                else:
                    region_nno[i] = [0, 0]  # all overlap
            else:
                if corners_t_s_min[i] > bbox_min_s[i]:
                    region_nno_0 = [corners_t_s_max[i], bbox_max_s[i]]
                    region_nno_1 = [bbox_min_s[i], corners_t_s_min[i]]
                    # Choose the one with larger non-overlap
                    if region_nno_0[1] - region_nno_0[0] > region_nno_1[1] - region_nno_1[0]:
                        region_nno[i] = region_nno_0
                    else:
                        region_nno[i] = region_nno_1
                else:
                    region_nno[i] = [corners_t_s_max[i], bbox_max_s[i]]
        # Compute the new bbox
        region_nno[0] = [bbox_min_s[0], bbox_max_s[0]]
        if proj_axis == "y":
            region_nno[1] = [bbox_min_s[1], bbox_max_s[1]]
        elif proj_axis == "z":
            region_nno[2] = [bbox_min_s[2], bbox_max_s[2]]
        bbox_extent_nno = region_nno[:, 1] - region_nno[:, 0]
        bbox_offset_nno = (region_nno[:, 1] + region_nno[:, 0]) / 2
        return bbox_offset_nno, bbox_extent_nno


def get_overlap_bbox_single(bbox_source, bbox_target, proj_axis: str = "z"):
    """We have a source bbox and a target bbox.
    FIXME: Currently, we use a very simple but coarse method to solve.
    proj_axis: the axis that we project the bbox to. "x", "y", "z", "none".
    """
    assert proj_axis == "z", "Only support z axis now."

    # Source
    bbox_center_s, bbox_rotvec_s, bbox_extent_s = bbox_source
    bbox_min_s = -bbox_extent_s / 2
    bbox_max_s = bbox_extent_s / 2
    bbox_rot_s = R.from_rotvec(bbox_rotvec_s).as_matrix()
    # Target
    bbox_center_t, bbox_rotvec_t, bbox_extent_t = bbox_target
    bbox_min_t = -bbox_extent_t / 2
    bbox_max_t = bbox_extent_t / 2
    bbox_rot_t = R.from_rotvec(bbox_rotvec_t).as_matrix()
    # Corners
    corners_s = get_corners(bbox_min_s, bbox_max_s, bbox_center_s, bbox_rot_s)
    corners_t = get_corners(bbox_min_t, bbox_max_t, bbox_center_t, bbox_rot_t)
    # corners_t in s frame
    corners_t_s = bbox_rot_s.T @ (corners_t - bbox_center_s).T
    corners_t_s_max = np.max(corners_t_s, axis=1)
    corners_t_s_min = np.min(corners_t_s, axis=1)
    # Compute the overlap part for each axis
    region_over = np.zeros((3, 2))
    for i in range(3):
        if corners_t_s_max[i] >= bbox_max_s[i]:
            if corners_t_s_min[i] >= bbox_max_s[i]:
                # No overlap
                region_over[i] = [0, 0]
            elif corners_t_s_min[i] > bbox_min_s[i]:
                region_over[i] = [corners_t_s_min[i], bbox_max_s[i]]
            else:
                region_over[i] = [bbox_min_s[i], bbox_max_s[i]]
        elif (corners_t_s_max[i] < bbox_max_s[i]) and (corners_t_s_max[i] > bbox_min_s[i]):
            if corners_t_s_min[i] > bbox_min_s[i]:
                region_over[i] = [corners_t_s_min[i], corners_t_s_max[i]]
            else:
                region_over[i] = [bbox_min_s[i], corners_t_s_max[i]]
        else:
            region_over[i] = [0, 0]  # No overlap
    # Compute the new bbox
    region_over[2] = [bbox_min_s[2], bbox_max_s[2]]
    bbox_extent_over = region_over[:, 1] - region_over[:, 0]
    bbox_offset_over = (region_over[:, 1] + region_over[:, 0]) / 2
    return bbox_offset_over, bbox_extent_over


################################## Object primitives ##################################


class LevelObject:
    def __init__(self, name, level):
        self.name = name
        self.level = level

    def __lt__(self, other):
        return self.level < other.level

    def __eq__(self, other):
        return (self.level == other.level) and (self.name == other.name)

    def __gt__(self, other):
        return self.level > other.level

    def __le__(self, other):
        return self.level <= other.level

    def __ge__(self, other):
        return self.level >= other.level

    def __ne__(self, other):
        return self.level != other.level


class Space:
    """Space is the container that used to store objects."""

    def __init__(
        self,
        name="",
        tf=np.eye(4),
        tf2parent=np.eye(4),
        extent=np.ones(3),
        space_type="drawer",
        dir_up=["+z"],
        trigger=[],
        is_visible=False,
        is_reachable=False,
    ):
        """Args:
        A standard space is defined by:
        up: z-axis, right: x-axis, forward: neg y-axis.
        In opengl,
        up: y-axis, right: x-axis, forward: z-axis.
        dir_up: direction of up in local coord
        space_type: drawer, shelf, open
        """
        self.name = name
        self.tf = np.array(tf)
        self.tf2parent = np.array(tf2parent)
        self.extent = np.array(extent)
        self.dir_up = dir_up  # Direction that is opened
        self.trigger = trigger
        self.is_visible = is_visible  # If this space is visible, currently it is the same as reachable
        self.is_reachable = is_reachable  # If this space is reachable
        self.contains = []  # Contained objects
        # For space structure, z really matters
        # Adjacency matrix, converting vector to standard coord
        self.adj_m = (
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ],
            )
            if "+y" in self.dir_up
            else np.eye(4)
        )
        # Compute opened dir in local coord
        # Open dir, pointing out to the opened direction
        self.space_type = space_type
        if space_type == "drawer":
            self.opened_dirs = [np.array([0, 0, 1])]
        elif space_type == "shelf":
            self.opened_dirs = [np.array([0, -1, 0])]
        elif space_type == "open":
            self.opened_dirs = [
                np.array([1, 0, 0]),
                np.array([-1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, -1, 0]),
                np.array([0, 0, 1]),
            ]
        else:
            raise NotImplementedError("Space type not implemented.")

    def get_cfg(self, name=""):
        cfg = {}
        cfg["name"] = self.name if name == "" else name
        cfg["extent"] = self.extent.tolist()
        cfg["type"] = "Space"
        cfg["tf"] = self.get_pose().tolist()
        cfg["is_visible"] = self.is_visible
        cfg["is_reachable"] = self.is_reachable

    def get_bbox(self):
        return np.array([self.tf[:3, 3], R.from_matrix(self.tf[:3, :3]).as_rotvec(), self.extent])

    def get_pose(self, is_raw=False, return_mat=False):
        rot_vec = R.from_matrix(self.tf[:3, :3]).as_rotvec()
        pose_6d = np.concatenate([self.tf[:3, 3], rot_vec])
        if is_raw:
            pose_6d[:3] -= self.tf[:3, :3] @ self.tf2parent[:3, 3]
        if return_mat:
            tf_mat = np.eye(4)
            tf_mat[:3, 3] = pose_6d[:3]
            tf_mat[:3, :3] = R.from_rotvec(pose_6d[3:]).as_matrix()
            return tf_mat
        return pose_6d

    def update_pose(self, pose):
        if pose.shape == (6,):
            self.tf[:3, 3] = pose[:3]
            self.tf[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
        elif pose.shape == (4, 4):
            self.tf = pose
            # Update all contains
            for obj in self.contains:
                obj.update_pose(self.tf @ obj.tf2parent)

    def sample_poses(
        self,
        object_extent,
        n_samples=1,
        rng=np.random.RandomState(0),
        enable_rot=False,
        ignore_size=False,
        place_bottom=True,
        angle_divide=2,
    ):
        if enable_rot and angle_divide > 1:
            angles = np.arange(0, np.pi / 2.0 + 1e-6, (np.pi / 2.0) / (angle_divide - 1))
        elif enable_rot and angle_divide == 1:
            angles = [0]
        else:
            angles = [0]
        pos_in_space_list = []
        for anlge in angles:
            local_rot = R.from_rotvec([0, 0, anlge]).as_matrix()
            pos_in_space = self.sample_poses_w_rot(
                object_extent,
                n_samples=n_samples,
                rng=rng,
                ignore_size=ignore_size,
                place_bottom=place_bottom,
                local_rot=local_rot,
            )
            pos_in_space_list.append(pos_in_space)
        return np.concatenate(pos_in_space_list, axis=0)

    def sample_poses_w_rot(
        self,
        object_extent,
        n_samples=1,
        rng=np.random.RandomState(0),
        ignore_size=False,
        place_bottom=True,
        local_rot=np.eye(3),
    ):
        """Sample pose for object with a given local orientation."""
        # Adj orientation
        extent_s = np.abs(self.adj_m[:3, :3] @ self.extent)
        # Compute extent after rot
        o_extent_r = np.abs(local_rot @ object_extent)
        o_extent_r -= 1e-6  # Avoid numerical issue
        pos_in_space_list = []
        if np.any(o_extent_r > extent_s):
            return []
        else:
            pos_in_space = rng.uniform(
                low=-extent_s[:3] / 2 + o_extent_r[:3] / 2,
                high=extent_s[:3] / 2 - o_extent_r[:3] / 2,
                size=(n_samples, 3),
            )
            pos_in_space_list.append(pos_in_space)

        if place_bottom:
            # Make z of object to at the bottom of the space.
            pos_in_space[:, 2] = -extent_s[2] / 2 + o_extent_r[2] / 2

        tf_in_space = np.zeros((n_samples, 4, 4))
        tf_in_space[:, :4, :4] = np.eye(4)
        tf_in_space[:, :3, :3] = local_rot
        tf_in_space[:, :3, 3] = pos_in_space
        tf_in_space = np.linalg.inv(self.adj_m) @ tf_in_space
        # Convert to world frame
        for i in range(n_samples):
            tf_in_space[i, :, :] = self.tf @ tf_in_space[i, :, :]
            # # Apply rot
            # # tf_in_space[i, :4, 3] =  self.tf @ self.adj_m @ tf_in_space[i, :4, 3]
            # tf_in_space[i, :3, :3] = self.tf[:3, :3] @ local_rot
        return tf_in_space

    def check_in(self, pos):
        """Check if a pos is in the space."""
        # TODO: currently do it a very naitve way
        if pos.shape == (4, 4):
            pos = pos[:3, 3]
        elif pos.shape == (6,):
            pos = pos[:3]
        elif pos.shape == (3,):
            pos = pos
        else:
            assert False, f"Pos should be (x, y, z), but get shape {pos.shape}"
        tf_inv = np.linalg.inv(self.tf)
        pos_local = pos @ tf_inv[:3, :3].T + tf_inv[:3, 3]
        if np.all(np.abs(pos_local) < self.extent / 2):
            return True
        else:
            return False

    def get_vis_o3d(self):
        import open3d as o3d

        bbox = o3d.geometry.OrientedBoundingBox(self.tf[:3, 3], self.tf[:3, :3], self.extent)
        bbox.color = np.array([0, 0, 1])

        # Space origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        origin.transform(self.tf)
        return [bbox, origin]


class Handle(Space):
    def __init__(
        self,
        name="",
        tf=np.eye(4),
        tf2parent=np.eye(4),
        extent=np.ones(3),
        space_type="drawer",
        dir_up=["+z"],
        trigger=[],
        is_visible=False,
        is_reachable=False,
    ):
        super().__init__(
            name=name,
            tf=tf,
            tf2parent=tf2parent,
            extent=extent,
            space_type=space_type,
            dir_up=dir_up,
            trigger=trigger,
            is_visible=is_visible,
            is_reachable=is_reachable,
        )
        # Compute the longer side apart from z
        longer_side_idx = np.argmax(self.extent[:2])
        self.longer_axis = np.array([[1, 0, 0], [0, 1, 0]][longer_side_idx])

    # An interactable region
    def get_vis_o3d(self):
        import open3d as o3d

        bbox = o3d.geometry.OrientedBoundingBox(self.tf[:3, 3], self.tf[:3, :3], self.extent)
        bbox.color = np.array([0, 1, 1])
        return [bbox]

    def get_grasp_poses(self, offset=0.1, strategy="top", bias=0.0):
        """Get the grasp pose for the handle. From z direction pointing to the handle.
        y axis of grasp pose should perpendicular to the longer side of handle.
        Args:
            offset: the offset for the grasp pose.
            grasp_strategy: top, side.
            bias: the bias for the grasp pose. negative for left, positive for right.
        Note: Accurately speaking, side is not a type of grasp, but use for pushing.
        """
        grasp_pose_local = np.eye(4)
        x_bias = self.extent[0] / 2 * bias * 0.8
        grasp_pose_local[:3, 3] = np.array([x_bias, 0, self.extent[2] / 2 + offset])
        rot = R.from_rotvec([0, np.pi, 0]).as_matrix()
        rot_y_axis = rot[:3, 1]
        if np.abs(np.dot(rot_y_axis, self.longer_axis)) > 0.99:
            # If they are parallel, we need to rotate 90 degree
            rot = rot @ R.from_rotvec([0, 0, np.pi / 2]).as_matrix()
            grasp_width = self.extent[0]  # Shorter side
        else:
            grasp_width = self.extent[1]
        grasp_pose_local[:3, :3] = rot
        grasp_pose = self.tf @ grasp_pose_local
        return [grasp_pose], [grasp_width]


class Link(LevelObject):
    """Link object."""

    def __init__(
        self,
        name: str,
        tf2parent=np.eye(4),
        extent=np.ones(3),
        tf=np.eye(4),
        parent=None,
        spaces=None,
        handles=None,
        scale=1.0,
        info=None,
    ):
        self.name = name
        self.parent = parent  # Parent, child is joint
        self.children = []
        self.tf = tf  # Pose w.r.t. world
        self.tf2parent = tf2parent  # Pose w.r.t. parent
        self.extent = extent
        self.level = parent.level + 1 if parent is not None else 0
        self.is_storage = True if spaces is not None else False
        self.scale = scale  # For logging reason; no real usage
        self.template_id = -1  # For manage path
        self.info = info
        # Add space
        if isinstance(spaces, list):
            self.spaces = {}
            for _idx, space in enumerate(spaces):
                self.spaces[f"{name}_{_idx}"] = space
        elif isinstance(spaces, dict):
            self.spaces = spaces  # A dict of spaces
        elif spaces is None:
            self.spaces = {}
        else:
            raise ValueError("Spaces has to be either list or dict or None...")
        # Add handle
        if isinstance(handles, list):
            self.handles = {}
            for _idx, handle in enumerate(handles):
                self.handles[f"{name}_{_idx}"] = handle
        elif isinstance(handles, dict):
            self.handles = handles  # A dict of handles
        elif handles is None:
            self.handles = {}
        else:
            raise ValueError("Handles has to be either list or dict or None...")
        super().__init__(self.name, self.level)

    def update_level(self, current_level):
        self.level = current_level
        for child in self.children:
            child.update_level(current_level + 1)

    def add_child(self, child):
        self.children.append(child)

    def add_space(self, space: Space):
        if space.name == "":
            space_name = f"{self.name}_space_{len(self.spaces)}"
        else:
            space_name = space.name
        if space_name in self.spaces:
            raise ValueError(f"Space {space_name} already exists.")
        self.spaces[space_name] = space

    def add_handle(self, handle: Handle):
        if handle.name == "":
            handle_name = f"{self.name}_handle_{len(self.handles)}"
        else:
            handle_name = handle.name
        if handle_name in self.handles:
            raise ValueError(f"Handle {handle_name} already exists.")
        self.handles[handle_name] = handle

    def update_pose(self, pose, is_raw=False):
        """Set pose.
        Args:
            pose: [x, y, z, rx, ry, rz]
            is_raw: if the pose is in raw form, i.e., without offset.
        """
        if pose.shape == (6,):
            self.tf[:3, 3] = pose[:3]
            self.tf[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
        elif pose.shape == (4, 4):
            self.tf = pose
        # TODO: Check if this is correct.
        if is_raw:
            self.tf[:3, 3] += self.tf[:3, :3] @ self.tf2parent[:3, 3]
        if self.spaces:
            for space in list(self.spaces.values()):
                space.update_pose(self.tf @ space.tf2parent)
        if self.handles:
            for handle in list(self.handles.values()):
                handle.update_pose(self.tf @ handle.tf2parent)

    def get_pose(self, is_raw=False, return_mat=False):
        rot_vec = R.from_matrix(self.tf[:3, :3]).as_rotvec()
        pose_6d = np.concatenate([self.tf[:3, 3], rot_vec])
        if is_raw:
            pose_6d[:3] -= self.tf[:3, :3] @ self.tf2parent[:3, 3]
        if return_mat:
            tf_mat = np.eye(4)
            tf_mat[:3, 3] = pose_6d[:3]
            tf_mat[:3, :3] = R.from_rotvec(pose_6d[3:]).as_matrix()
            return tf_mat
        return pose_6d

    def get_bbox(self):
        return np.array([self.tf[:3, 3], R.from_matrix(self.tf[:3, :3]).as_rotvec(), self.extent])

    def get_vis_o3d(self):
        import open3d as o3d

        bbox = o3d.geometry.OrientedBoundingBox(self.tf[:3, 3], self.tf[:3, :3], self.extent)
        bbox.color = np.array([1, 0, 0])
        vis = [bbox]
        # show origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        origin.transform(self.tf)
        vis.append(origin)

        # show raw origin
        raw_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        raw_tf = self.get_pose(is_raw=True, return_mat=True)
        raw_origin.transform(raw_tf)
        vis.append(raw_origin)
        # [DEBUG]: show handles
        if self.handles:
            for handle in list(self.handles.values()):
                bbox_handle = handle.get_vis_o3d()
                vis += bbox_handle
        return vis


class RigidObject(Link):
    """Rigid object's center can have an offset with the bbox center."""

    def __init__(
        self,
        name: str,
        tf2parent=np.eye(4),
        extent=np.ones(3),
        tf=np.eye(4),
        offset=np.zeros(3),
        parent=None,
        spaces=None,
        handles=None,
        scale=1.0,
        is_visible=False,
        is_reachable=False,
        info=None,
    ):
        tf2parent[:3, 3] += offset  # apply offset
        super().__init__(
            name=name,
            tf2parent=tf2parent,
            extent=extent,
            tf=tf,
            parent=parent,
            spaces=spaces,
            handles=handles,
            scale=scale,
            info=info,
        )
        self.offset = offset
        self.is_visible = is_visible
        self.is_reachable = is_reachable
        # Compute the longer side apart from z for grasp
        longer_side_idx = np.argmax(self.extent[:2])
        self.longer_axis = np.array([[1, 0, 0], [0, 1, 0]][longer_side_idx])

    def get_cfg(self):
        cfg = {}
        cfg["name"] = self.name
        cfg["type"] = "RigidObject"
        cfg["load_from_template"] = self.template_id
        cfg["tf"] = self.get_pose().tolist()
        cfg["tf_raw"] = self.get_pose(is_raw=True).tolist()
        cfg["scale"] = self.scale
        cfg["extent"] = self.extent.tolist()
        return cfg

    def get_vis_o3d(self):
        import open3d as o3d

        vis_o3ds = super().get_vis_o3d()
        for vis_o3d in vis_o3ds:
            if isinstance(vis_o3d, o3d.geometry.OrientedBoundingBox):
                vis_o3d.color = np.array([0, 1, 0])
            # elif isinstance(vis_o3d, o3d.geometry.TriangleMesh):
            #     vis_o3d.paint_uniform_color([0, 1, 0])
        raw_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        raw_origin.transform(self.tf)
        raw_origin.translate(-self.offset)
        vis_o3ds.append(raw_origin)
        return vis_o3ds

    def get_pose(self, is_raw=False, return_mat=False):
        # Interact with extenral interface with raw pose
        pose = super().get_pose(is_raw=is_raw, return_mat=return_mat)
        return pose

    def update_pose(self, pose, is_raw=False):
        # Interact with extenral interface with raw pose
        if is_raw:
            if pose.shape == (6,):
                pose[:3] += self.offset
            elif pose.shape == (4, 4):
                pose[:3, 3] += self.offset
            else:
                raise NotImplementedError("Shape not implemented.")
        super().update_pose(pose)

    def get_grasp_poses(self, offset=0.1, by_bbox=True, wrt_obj=False, strategy="top", bias=0.0):
        """Generate multiple grasp poses.
        by_bbox: get the grasp pose as the object is a bbox.
        strategy: top, side, all.
        """
        # FIXME: this assume the object's z-axis is pointing up.
        # May not be true for more general cases.
        assert by_bbox is True, "Only support by_bbox for now."
        grasp_poses = []
        grasp_widths = []

        ## BBox primitives, from top, side
        # Top-down
        if strategy == "top" or strategy == "all":
            grasp_pose_local = np.eye(4)
            x_bias = self.extent[0] / 2 * bias * 0.8
            grasp_pose_local[:3, 3] = np.array([x_bias, 0, self.extent[2] / 2 + offset])
            rot = R.from_rotvec([0, np.pi, 0]).as_matrix()
            rot_y_axis = rot[:3, 1]
            if np.abs(np.dot(rot_y_axis, self.longer_axis)) > 0.99:
                # If they are parallel, we need to rotate 90 degree
                rot = rot @ R.from_rotvec([0, 0, np.pi / 2]).as_matrix()
                grasp_width = self.extent[0]  # Shorter side
            else:
                grasp_width = self.extent[1]  # Shorter side
            grasp_pose_local[:3, :3] = rot
            grasp_poses.append(grasp_pose_local)
            grasp_widths.append(grasp_width)

        # From side, grasp by the shorter side of x or y
        if strategy == "side" or strategy == "all":
            grasp_pose_local = np.eye(4)
            rot = R.from_rotvec([0, np.pi, 0]).as_matrix()
            z_bias = self.extent[2] / 2 * bias * 0.8
            if self.longer_axis[1] > 0.99:
                # y is the longer side, grasp from y direction
                grasp_pose_local[:3, 3] = np.array([0, self.extent[1] / 2 + offset, z_bias])
                grasp_pose_local[:3, :3] = R.from_rotvec([0, -np.pi / 2, 0]).as_matrix() @ R.from_rotvec([np.pi / 2.0, 0, 0]).as_matrix()
                grasp_width = self.extent[0]
            else:
                # x is the longer side, grasp from x direction
                grasp_pose_local[:3, 3] = np.array([self.extent[0] / 2 + offset, 0, z_bias])
                grasp_pose_local[:3, :3] = rot @ R.from_rotvec([-np.pi / 2, 0, 0]).as_matrix()
                grasp_width = self.extent[1]

            grasp_poses.append(grasp_pose_local)
            grasp_widths.append(grasp_width)

        if not wrt_obj:
            grasp_poses = [self.tf @ x for x in grasp_poses]
        return grasp_poses, grasp_widths


class Joint(LevelObject):
    """Joint object."""

    def __init__(
        self,
        name: str,
        joint_type: str,
        parent: Link,
        tf2parent: np.ndarray,
        axis_local=np.array([0, 0, 0]),
        limits=np.array([0, 1]),
        child: Link = None,
        scale=1.0,
        info=None,
    ):
        self.name = name
        self.child = child
        self.parent = parent
        self.joint_type = joint_type
        self.joint_value = 0.0
        self.tf = np.eye(4)
        self.tf_local = np.eye(4)
        self.tf2parent = tf2parent
        self.axis_local = axis_local  # Articulation axis in local frame
        self.limits = limits
        self.level = parent.level  # Joint level
        self.scale = scale
        self.info = info
        super().__init__(self.name, self.level)

    def update_level(self, current_level):
        self.level = current_level
        if self.child is not None:
            self.child.update_level(current_level + 1)

    def set_child(self, child):
        self.child = child

    def set_joint_value(self, value, is_rel=True):
        """Radius value for revolute joint."""
        if not is_rel:
            self.joint_value = value
        else:
            self.joint_value = self.limits[0] + value * (self.limits[1] - self.limits[0])
        self._update_pose()

    def get_axis(self):
        """Get the dir & origin of the axis in world frame."""
        return self.tf[:3, :3] @ self.axis_local, self.tf[:3, 3]

    def _update_pose(self):
        if self.joint_type == "revolute":
            self.tf_local[:3, :3] = np.eye(3)
            self.tf_local[:3, :3] = R.from_rotvec(self.axis_local * self.joint_value).as_matrix()
        elif self.joint_type == "prismatic":
            self.tf_local[:3, 3] = self.axis_local * self.joint_value
        elif self.joint_type == "rigid" or self.joint_type == "fixed":
            self.tf_local = np.eye(4)
        else:
            raise NotImplementedError("Joint type not implemented.")


class ArtObject(Link):
    """Abstract class for articulation object."""

    def __init__(
        self,
        name: str = "",
        extent=np.ones(3),
        tf=np.eye(4),
        spaces=None,
        scale=1.0,
        info=None,
    ):
        # Global scale, influencing the global scale for this art object.
        super().__init__(name=name, extent=extent, tf=tf, spaces=spaces, scale=scale)
        self.name = name
        self.obj_name = ""
        self.art_tree = None
        self.links = {name: self}
        self.joints = {}
        self.active_joints = []
        self.spaces = {}
        self.handles = {}
        if spaces is not None:
            for _idx, space in enumerate(spaces):
                self.spaces[f"{name}_{_idx}"] = space
        else:
            self.spaces = {}
        self.info = info

    ############################### Macro methods ###############################
    def get_cfg(self):
        cfg = {}
        cfg["name"] = self.name
        cfg["type"] = "ArtObject"
        cfg["load_from_template"] = self.template_id
        cfg["tf"] = self.get_pose().tolist()
        cfg["tf_raw"] = self.get_pose(is_raw=True).tolist()
        cfg["joint_values"] = self.get_joint_values().tolist()
        cfg["scale"] = self.scale
        cfg["extent"] = self.extent.tolist()
        return cfg

    def add_link(
        self,
        name,
        tf2parent=np.eye(4),
        extent=np.ones(3),
        parent: Joint = None,
        spaces: Space | list[Space] = None,
        handles: Handle | list[Handle] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        assert name not in self.links, "Link already exists."
        if isinstance(parent, str):
            parent = self.joints.get(parent, None)
        self.links[name] = Link(
            name=name,
            tf2parent=tf2parent,
            extent=extent,
            parent=parent,
            spaces=spaces,
            handles=handles,
            scale=scale,
            info=kwargs.get("info", None),
        )
        if parent is not None and parent.child is None:
            parent.set_child(self.links[name])
        if spaces is not None:
            for _idx, space in enumerate(spaces):
                space_name = space.name if space.name != "" else f"{name}_space_{_idx}"
                self.spaces[space_name] = space  # Reference to space
        if handles is not None:
            for _idx, handle in enumerate(handles):
                handle_name = handle.name if (handle.name != "handle" and handle.name != "") else f"{name}_handle_{_idx}"
                self.handles[handle_name] = handle
        return self.links[name]

    def add_joint(
        self,
        name: str,
        joint_type: str,
        parent: Link | str,
        tf2parent: np.ndarray,
        axis_local: np.ndarray,
        limits: np.ndarray = np.array([0, 1]),
        child: Link | str = None,
        scale: float = 1.0,
        info=None,
    ):
        assert name not in self.joints, "Joint already exists."
        if isinstance(parent, str):
            parent = self.links.get(parent, None)
        if isinstance(child, str):
            child = self.links.get(child, None)
        self.joints[name] = Joint(name, joint_type, parent, tf2parent, axis_local, limits, child, scale, info)
        if joint_type in ["revolute", "prismatic"]:
            self.active_joints.append(name)
        if child is not None:
            child.parent = self.joints[name]
        parent.add_child(self.joints[name])
        return self.joints[name]

    ############################### Low-level methods ###############################
    def process_trigger(self):
        # Run space triggers
        joint_values = self.get_joint_values(is_rel=True, return_dict=True)
        for name, space in self.spaces.items():
            triggered = True
            # FIXME: double checking this trigger
            if len(space.trigger) == 0:
                # No trigger closed space
                triggered = False
            else:
                for trigger_data in space.trigger:
                    if len(trigger_data) == 0:
                        # Empty trigger, closed space
                        triggered = False
                    elif len(trigger_data) == 2:
                        trigger_name, trigger_value = trigger_data
                        if trigger_name == "all":
                            triggered = True
                            break  # Open space
                        if (trigger_name not in joint_values) or (joint_values[trigger_name] < trigger_value):
                            triggered = False
            if triggered:
                # Update attrib
                space.is_reachable = True
                space.is_visible = True
            else:
                space.is_reachable = False
                space.is_visible = False

    def update_level(self):
        # Update level recursively
        for child in self.children:
            child.update_level(1)

    def remove_link(self, link_name):
        """Remove link. And remove the joint if no other links are connected."""
        link = self.links[link_name]
        if link.parent is not None:
            joint = link.parent
            if len(joint.parent.children) == 1:
                joint.child = None
                del self.joints[joint.name]
        del self.links[link_name]

    def forward_pose(self):
        """Forward kinematics."""
        self.process_trigger()  # Run trigger first
        # For low level to high level
        for joint in sorted(self.joints.values()):
            joint._update_pose()
            # joint's pose w.r.t to last joint's pose
            if joint.parent.parent is not None:
                parent_tf = joint.parent.parent.tf @ joint.parent.parent.tf_local
            elif joint.parent.name == "base":
                parent_tf = self.tf
            else:
                parent_tf = np.eye(4)
            joint.tf = parent_tf @ joint.tf2parent
            joint_local_tf = joint.tf @ joint.tf_local
            if joint.child is not None:
                joint.child.update_pose(joint_local_tf @ joint.child.tf2parent)

    def update_pose(self, pose, move_center=True):
        """There is an offset between art-center, and base link.
        So in order to mitgate it, we need to provide an additional shift.
        """
        # Override update pose
        pose_base = deepcopy(pose)
        if move_center:
            bbox_min, bbox_max = self.get_whole_bbox()
            bbox_center = (bbox_max + bbox_min) / 2.0
            base_pos = self.tf[:3, 3]
            center_offset = bbox_center - base_pos
            if pose.shape == (6,):
                pose_base[:3] -= center_offset
            elif pose.shape == (4, 4):
                pose_base[:3, 3] -= center_offset
            else:
                raise NotImplementedError("Shape not implemented.")
        super().update_pose(pose_base)
        self.forward_pose()

    def set_joint_value(self, joint_name, value, is_rel=True):
        """Set joint value."""
        self.joints[joint_name].set_joint_value(value, is_rel)
        # Update pose after change value
        self.forward_pose()

    def set_joint_values(self, joint_values: dict | np.ndarray | list | float, is_rel=True):
        """Set joint values."""
        # FIXME: We can only set activate nodes
        if isinstance(joint_values, dict):
            for joint_name, value in joint_values.items():
                self.joints[joint_name].set_joint_value(value, is_rel)
        elif isinstance(joint_values, np.ndarray) or isinstance(joint_values, list):
            if isinstance(joint_values, np.ndarray):
                if joint_values.shape == ():
                    joint_values = [joint_values] * len(self.active_joints)
            assert len(joint_values) == len(self.active_joints), "Joint values should equal active joints."
            for idx, joint_name in enumerate(self.active_joints):
                self.joints[joint_name].set_joint_value(joint_values[idx], is_rel)
        elif isinstance(joint_values, float) or isinstance(joint_values, int):
            for idx, joint_name in enumerate(self.active_joints):
                self.joints[joint_name].set_joint_value(joint_values, is_rel)
        else:
            raise ValueError("Joint value type not recognized.")
        # Update pose after change value
        self.forward_pose()

    def get_joint_values(self, is_rel=True, return_dict=False, in_isaac_order=False):
        """Get joint values.
        in_isaac_order: if the joint values are in isaac order.
        """
        real_joint_values = np.array([self.joints[joint_name].joint_value for joint_name in self.active_joints])
        if is_rel:
            joint_values = np.array(
                [
                    (self.joints[joint_name].joint_value - self.joints[joint_name].limits[0])
                    / (self.joints[joint_name].limits[1] - self.joints[joint_name].limits[0])
                    for joint_name in self.active_joints
                ]
            )
        else:
            joint_values = np.array([self.joints[joint_name].joint_value for joint_name in self.active_joints])
        if in_isaac_order:
            # Convert to isaac order
            isaac_joint_values = np.zeros(len(joint_values))
            for isaac_idx, lgmcts_idx in self.get_isaac2lgmcts_idx().items():
                isaac_joint_values[isaac_idx] = joint_values[lgmcts_idx]
            joint_values = isaac_joint_values
        if not return_dict:
            return joint_values
        else:
            joint_value_dict = {}
            for idx, joint_name in enumerate(self.active_joints):
                joint_value_dict[joint_name] = joint_values[idx]
            return joint_value_dict

    def get_joint_types(self):
        """Get joint types."""
        return [self.joints[joint_name].joint_type for joint_name in self.active_joints]

    def get_joint_raw_names(self):
        """Get joint raw names."""
        return [self.joints[joint_name].info.get("raw_name", "") for joint_name in self.active_joints]

    def check_storage(self, link_name: str | int):
        if isinstance(link_name, int):
            return list(self.links.values())[link_name].is_storage
        elif isinstance(link_name, str):
            return self.links[link_name].is_storage
        else:
            raise ValueError("Invalid input.")

    def get_bboxes(self, block_list=[]):
        """Get bounding boxes for checking."""
        self.forward_pose()
        bboxes = []
        for link in self.links.values():
            if link.name == "base" or link.name in block_list:
                # Skip base, as it is not a real link
                continue
            bbox_array = np.array(
                [
                    link.tf[:3, 3],
                    R.from_matrix(link.tf[:3, :3]).as_rotvec(),
                    link.extent,
                ]
            )
            bboxes.append(bbox_array)
        return bboxes

    ######################### Motion planning related #########################
    def get_manip_traj(
        self,
        joint_idx: str | int,
        goal_joint_value,
        num_waypoints=50,
        offset=0.1,
        pre_offset=0.2,
        pre_num_waypoints=10,
        move_over_head=0.1,
        bias=0.0,
        grasp_pose=None,
        grasp_width=None,
    ):
        """Generate the manipulation trajectory for a joint.
        Args:
            bias: bias for grasping. When bias is 0, grasp at center. bias > 0, grasping towards right. bias < 0, grasping towards left.
        """
        # FIXME: May have some bugs.
        if isinstance(joint_idx, str):
            assert joint_idx in self.active_joints, "Joint not in active joints."
            joint_idx = self.active_joints.index(joint_idx)
        assert joint_idx < len(self.active_joints), "Joint index out of range."
        joint = self.joints[self.active_joints[joint_idx]]
        link = joint.child
        if not link.handles:
            print("No handle for this link.")
            return None, None, 0.0, "Fail"
        # Choose the handle biggest handle.
        handle_size = [max(handle.extent[0], handle.extent[1]) for handle in list(link.handles.values())]
        handle_idx = np.argmax(handle_size)
        handle = list(link.handles.values())[handle_idx]

        cur_joint_value = self.get_joint_values()[joint_idx]
        joint_values = np.linspace(cur_joint_value, goal_joint_value, num_waypoints)
        traj = []
        # Compute grasp pose first
        grasp_pose_local = None  # Grasp pose in local frame
        if grasp_pose is None or grasp_width is None:
            grasp_poses, grasp_widths = handle.get_grasp_poses(offset=offset, bias=bias)
            grasp_pose = grasp_poses[0]
            grasp_width = grasp_widths[0]
        else:
            pass
        if grasp_pose_local is None:
            grasp_pose_local = np.linalg.inv(handle.tf) @ grasp_pose

        for idx in range(num_waypoints + 2 * pre_num_waypoints):
            if idx <= pre_num_waypoints:
                cur_offset = np.interp(idx, [0, pre_num_waypoints], [pre_offset, offset])
                # pre-grasp pose
                joint_value = joint_values[0]
            elif idx >= (num_waypoints + pre_num_waypoints):
                # post-grasp pose
                cur_offset = (
                    np.interp(
                        idx - (num_waypoints + pre_num_waypoints),
                        [0, pre_num_waypoints],
                        [offset, pre_offset],
                    )
                    + move_over_head
                )
                joint_value = joint_values[-1]
            else:
                # move joint
                joint_value = joint_values[idx - pre_num_waypoints]
                # gradually improve overhead
                _move_over_head = np.interp((idx - pre_num_waypoints), [0, num_waypoints], [0, move_over_head])
                cur_offset = offset + _move_over_head
            joint.set_joint_value(joint_value, is_rel=True)
            self.forward_pose()
            # Compute ee pose
            grasp_pose_local_cur = np.copy(grasp_pose_local)
            grasp_pose_local_cur[2, 3] = handle.extent[2] / 2.0 + cur_offset  # z-value
            ee_pose = handle.tf @ grasp_pose_local_cur
            traj.append(ee_pose)
        # Reset joint value
        joint.set_joint_value(cur_joint_value, is_rel=True)
        self.forward_pose()
        return traj, grasp_pose, grasp_width, "Success"

    ######################### Utility functions #########################
    def get_spaces(self, exclude: list[str] = [], size_threshold=0.06):
        """Get all spaces except for the excluded ones. And filter out small spaces."""
        spaces = {}
        for space_name, space in self.spaces.items():
            if not any([ex in space_name for ex in exclude]):
                spaces[space_name] = space
        # Filter out small spaces
        if size_threshold > 0.0:
            spaces = {space_name: space for space_name, space in spaces.items() if min(space.extent) > size_threshold}
        return spaces

    def get_isaac2lgmcts_idx(self):
        # Joint mapping between isaac and lgmcts
        # isaac gym order joints by their raw names
        # lgmcts order joints by heirachical orders
        # This matters when we have more than 10 joints
        isaac2lgmcts_idx = {}
        raw_names = []
        for joint_name in self.active_joints:
            link = self.joints[joint_name].child
            raw_name = link.info.get("raw_name", "")
            if raw_name != "":
                raw_names.append(raw_name)
        raw_names = sorted(raw_names)
        for idx, joint_name in enumerate(self.active_joints):
            link = self.joints[joint_name].child
            link_raw_name = link.info.get("raw_name", "")
            isaac2lgmcts_idx[raw_names.index(link_raw_name)] = idx
        return isaac2lgmcts_idx

    def get_joint_idx(self, joint_name):
        for idx, _joint_name in enumerate(self.active_joints):
            if _joint_name == joint_name:
                return idx
        return None

    def get_joint_limits(self, return_dict=False):
        """Get joint limits."""
        joint_limits = np.array([self.joints[joint_name].limits for joint_name in self.active_joints])
        if return_dict:
            joint_limit_dict = {}
            for idx, joint_name in enumerate(self.active_joints):
                joint_limit_dict[joint_name] = joint_limits[idx]
            return joint_limit_dict
        else:
            return joint_limits

    def get_whole_bbox(self, block_list: list[str] = []):
        """Get the bbox for the whole object."""
        bboxes = self.get_bboxes(block_list)
        # Get corners
        corner_points = []
        for bbox_arr in bboxes:
            bbox_center, bbox_rotvec, bbox_extent = bbox_arr
            bbox_min = -bbox_extent / 2
            bbox_max = bbox_extent / 2
            bbox_rot = R.from_rotvec(bbox_rotvec).as_matrix()
            bbox_corners = get_corners(bbox_min, bbox_max, bbox_center, bbox_rot)
            corner_points.append(bbox_corners)
        corner_points = np.stack(corner_points).reshape(-1, 3)
        bbox_min = np.min(corner_points, axis=0)
        bbox_max = np.max(corner_points, axis=0)
        return bbox_min, bbox_max

    def get_contained_objs(self):
        """Get contained objects."""
        contained_objs = []
        for space in self.spaces.values():
            for obj in space.contains:
                if isinstance(obj, ArtObject):
                    contained_objs += obj.get_contained_objs()
                elif isinstance(obj, RigidObject):
                    contained_objs.append(obj)
                    if obj.spaces is not None:
                        for obj_space in obj.spaces.values():
                            contained_objs += obj_space.contains
        return contained_objs

    def get_storage_properties(self):
        """Get storage property."""
        storage_properties = []
        for link in self.links.values():
            if link.name == "base":
                # Skip base, as it is not a real link
                continue
            if link.is_storage:
                storage_properties.append(True)
            else:
                storage_properties.append(False)
        return storage_properties

    def get_vis_o3d(self, show_joint=False):
        """Visualize articulation structure in tree."""
        import open3d as o3d

        self.forward_pose()
        link_vis = []
        for link in self.links.values():
            link_bbox = o3d.geometry.OrientedBoundingBox(link.tf[:3, 3], link.tf[:3, :3], link.extent)
            # if link.is_storage:
            #     # Add space vis
            #     space_bbox = self.spaces[link.name].get_vis_o3d()
            #     link_vis += space_bbox
            #     color = np.array([0, 1, 0])
            color = np.array([1, 0, 0])
            link_bbox.color = color
            link_vis.append(link_bbox)
            # [Show handle]
            if link.handles:
                for handle in list(link.handles.values()):
                    link_vis += handle.get_vis_o3d()

        joint_vis = []
        for joint in self.joints.values():
            joint_axis = get_arrow(joint.axis_local, origin=np.zeros(3), scale=0.1)
            joint_axis.paint_uniform_color([1, 0, 0])
            joint_axis.transform(joint.tf)
            joint_vis.append(joint_axis)

        # [DEBUG]: Show the triggers
        trigger_vis = []
        for space in self.spaces.values():
            space_center = space.tf[:3, 3]
            for trigger_data in space.trigger:
                if len(trigger_data) == 2:
                    trigger_name, trigger_value = trigger_data
                    trigger_link = self.links.get(trigger_name, None)
                    if trigger_link is not None:
                        trigger_center = trigger_link.tf[:3, 3]
                        # Draw line
                        line = o3d.geometry.LineSet()
                        line.points = o3d.utility.Vector3dVector([space_center, trigger_center])
                        line.lines = o3d.utility.Vector2iVector([[0, 1]])
                        line.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
                        trigger_vis.append(line)

        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        if not show_joint:
            return link_vis + trigger_vis
        else:
            return link_vis + joint_vis + trigger_vis


class Robot(ArtObject):
    def __init__(
        self,
        name: str = "",
        extent=np.ones(3),
        tf=np.eye(4),
        spaces=None,
        scale=1.0,
        info=None,
    ):
        super().__init__(name=name, extent=extent, tf=tf, spaces=spaces, scale=scale, info=info)
        self.ee_name = info.get("ee_name", "ee")

    def get_ee_pose(self):
        return self.links[self.ee_name].tf

    def get_ee_link(self):
        return self.links[self.ee_name]

    def get_vis_o3d(self, show_joint=False):
        import open3d as o3d

        vis_o3d = super().get_vis_o3d(show_joint)
        # Set purple color for robot
        for vis in vis_o3d:
            vis.color = np.array([1, 0, 1])
        # Add ee tf
        ee_tf = self.links[self.ee_name].tf
        ee_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee_origin.transform(ee_tf)
        vis_o3d.append(ee_origin)
        return vis_o3d


################################## Manip method ##################################
def get_grasp_traj(grasp_pose, offset, grasp_waypoints):
    pre_grasp_pose = grasp_pose.copy()
    pre_grasp_pose[:3, 3] -= offset * grasp_pose[:3, 2]
    grasp_traj = []
    grasp_traj.append(pre_grasp_pose)
    for i in range(grasp_waypoints):
        wp_pose = pre_grasp_pose.copy()
        wp_pose[:3, 3] += (i + 1) / grasp_waypoints * offset * grasp_pose[:3, 2]
        grasp_traj.append(wp_pose)
    grasp_traj.append(grasp_pose)
    return grasp_traj


def sample_poses_with_gripper(
    space,
    obj_ext,
    gripper_ext,
    grasp_pose_local,
    n_samples=1,
    rng=np.random.RandomState(0),
    enable_rot=False,
    place_bottom=True,
    angle_divide=2,
    # [DEBUG]
    object=None,
    gripper=None,
    art_obj=None,
    debug=False,
):
    """Sample a pose providing a grasp pose, and grasp robot."""
    # First compute the combined bbox
    gripper_bdary = np.array(
        [
            [gripper_ext[0] / 2.0, gripper_ext[1] / 2.0, gripper_ext[2] / 2.0],
            [-gripper_ext[0] / 2.0, -gripper_ext[1] / 2.0, -gripper_ext[2] / 2.0],
        ]
    )
    gripper_bdary = np.dot(gripper_bdary, grasp_pose_local[:3, :3].T) + grasp_pose_local[:3, 3]
    object_bdary = np.array(
        [
            [obj_ext[0] / 2.0, obj_ext[1] / 2.0, obj_ext[2] / 2.0],
            [-obj_ext[0] / 2.0, -obj_ext[1] / 2.0, -obj_ext[2] / 2.0],
        ]
    )
    comb_bdary = np.concatenate([gripper_bdary, object_bdary], axis=0)
    comb_ext = np.max(comb_bdary, axis=0) - np.min(comb_bdary, axis=0)
    # From the center of combined bbox to center of obj
    comb_off = (np.max(comb_bdary, axis=0) + np.min(comb_bdary, axis=0)) / 2
    # Sample a pose for the cmb_ext
    # The direction of gripper in the world coordinate must be aligned with an open direction
    gripper_dir = -grasp_pose_local[:3, 2]
    # gripper_release_dir = grasp_pose_local[:3, 0]
    opened_dirs = space.opened_dirs
    gravity_dir = np.array([0, 0, -1])
    obj_poses_candid = []
    # Local rotation along the opened_dir
    if enable_rot and angle_divide > 1:
        angles = np.arange(0, np.pi / 2.0 + 1e-6, (np.pi / 2.0) / (angle_divide - 1))
    elif enable_rot and angle_divide == 1:
        angles = [0]
    else:
        angles = [0]

    for opened_dir in opened_dirs:
        # Compute the local_rot for obj, local_rot @ gripper_dir = opened_dir
        # Also, for grasping structure,the y-axis of gripper should not be aligned with gravity
        # FIXME: my current computation may not be optimal
        cos_theta = np.dot(opened_dir, gripper_dir)
        if cos_theta > 0.99:
            local_rot = np.eye(3)
        else:
            theta = np.arccos(cos_theta)
            axis = np.cross(opened_dir, gripper_dir)
            axis /= np.linalg.norm(axis)
            local_rot = R.from_rotvec(theta * axis).as_matrix()
            # Check direction
            if not np.allclose(local_rot @ gripper_dir, opened_dir, atol=1e-3):
                local_rot = R.from_rotvec(-theta * axis).as_matrix()
            # Verity side direction
            gripper_y_axis = local_rot @ grasp_pose_local[:3, 1]
            if np.abs(np.dot(gripper_y_axis, gravity_dir)) > 0.9:
                # If they are parallel, we need to rotate 90 degree
                local_rot = local_rot @ R.from_rotvec([0, 0, np.pi / 2]).as_matrix()
        # Expend the space by the gripper size along open_dir
        space_ext = deepcopy(space.extent)
        space_ext += np.abs(np.linalg.inv(space.adj_m[:3, :3]) @ opened_dir) * gripper_ext[2]
        space_off = np.abs(np.linalg.inv(space.adj_m[:3, :3]) @ opened_dir) * gripper_ext[2] / 2
        space_tf = deepcopy(space.tf)
        space_tf[:3, 3] += space.tf[:3, :3] @ space_off
        space_exp = Space(
            tf=space_tf,
            tf2parent=space.tf2parent,
            extent=space_ext,
            dir_up=space.dir_up,
            trigger=space.trigger,
            is_visible=space.is_visible,
            is_reachable=space.is_reachable,
        )
        # Sample pose
        obj_poses = []
        for angle in angles:
            # Rotate the local_rot
            local_rot_rot = local_rot @ R.from_rotvec(gripper_dir * angle).as_matrix()
            _obj_poses = space_exp.sample_poses_w_rot(
                object_extent=comb_ext,
                n_samples=n_samples,
                rng=np.random.RandomState(0),
                ignore_size=False,
                place_bottom=place_bottom,
                local_rot=local_rot_rot,
            )
            if len(_obj_poses) > 0:
                obj_poses.append(_obj_poses)
        if len(obj_poses) == 0:
            continue
        obj_poses = np.concatenate(obj_poses, axis=0)

        for _i in range(len(obj_poses)):
            real_obj_pose = deepcopy(obj_poses[_i])
            real_obj_pose[:3, 3] -= real_obj_pose[:3, :3] @ comb_off
            obj_poses_candid.append(real_obj_pose)

            # [DEBUG]
            # debug = True
            if debug and art_obj is not None and object is not None and gripper is not None:
                # [DEBUG]
                import open3d as o3d

                vis = []
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                vis += [origin]
                vis += art_obj.get_vis_o3d()
                # comb link
                comb_link = Link("comb", extent=comb_ext)
                comb_link.update_pose(obj_poses[_i])
                vis += comb_link.get_vis_o3d()
                vis[-3].color = np.array([0, 1, 1])
                vis += space_exp.get_vis_o3d()
                vis += space.get_vis_o3d()
                object.update_pose(real_obj_pose)
                gripper.update_pose(real_obj_pose @ grasp_pose_local)
                vis += object.get_vis_o3d()
                vis += gripper.get_vis_o3d()
                o3d.visualization.draw_geometries(vis)

    if len(obj_poses_candid) == 0:
        return []
    # Select from the candidates
    select_idx = rng.choice(list(range(len(obj_poses_candid))), n_samples, replace=False)
    obj_poses = [obj_poses_candid[_i] for _i in select_idx]
    return obj_poses


################################## Initialize method ##################################
def load_space_from_config(config, scale=1.0):
    tf_6d = np.array(config.get("tf", [0, 0, 0, 0, 0, 0]))
    tf = np.eye(4)
    tf[:3, 3] = tf_6d[:3] * scale
    tf[:3, :3] = R.from_rotvec(tf_6d[3:]).as_matrix()

    tf2parent_6d = np.array(config.get("tf2parent", [0, 0, 0, 0, 0, 0]))
    tf2parent = np.eye(4)
    tf2parent[:3, 3] = tf2parent_6d[:3] * scale
    tf2parent[:3, :3] = R.from_rotvec(tf2parent_6d[3:]).as_matrix()

    extent = np.array(config["extent"]) * scale  # Must be defined
    dir_up = config.get("dir_up", ["+z"])
    trigger = config.get("trigger", [])
    is_visible = config.get("is_visible", False)
    is_reachable = config.get("is_reachable", False)
    space_type = config.get("space_type", "drawer")
    return {
        "name": config.get("name", "space"),
        "tf": tf,
        "tf2parent": tf2parent,
        "extent": extent,
        "dir_up": dir_up,
        "trigger": trigger,
        "space_type": space_type,
        "is_visible": is_visible,
        "is_reachable": is_reachable,
    }


def load_handle_from_config(config, scale=1.0):
    handle_dict = load_space_from_config(config, scale=scale)
    handle_dict["name"] = config.get("name", "handle")
    return handle_dict


def load_link_from_config(config, scale=1.0):
    link_name = config["name"]
    tf6d = np.array(config.get("tf", [0, 0, 0, 0, 0, 0]))
    tf = np.eye(4)
    tf[:3, 3] = tf6d[:3] * scale  # Apply scale to trans
    tf[:3, :3] = R.from_rotvec(tf6d[3:]).as_matrix()

    tf2parent_6d = np.array(config.get("tf2parent", [0, 0, 0, 0, 0, 0]))
    tf2parent = np.eye(4)
    tf2parent[:3, 3] = tf2parent_6d[:3] * scale  # Apply scale to trans
    tf2parent[:3, :3] = R.from_rotvec(tf2parent_6d[3:]).as_matrix()
    extent = np.array(config["extent"]) * scale  # Apply scale to extent
    parent_name = config.get("parent", None)
    offset = np.array(config.get("offset", [0, 0, 0])) * scale
    # Add spaces
    if "spaces" in config:
        spaces = [Space(**load_space_from_config(s, scale=scale)) for s in config["spaces"]]
    else:
        spaces = None
    # Add handles
    if "handles" in config:
        handles = [Handle(**load_handle_from_config(h, scale=scale)) for h in config["handles"]]
    else:
        handles = None
    # Add info
    info = config.get("info", None)
    return {
        "name": link_name,
        "tf": tf,
        "tf2parent": tf2parent,
        "extent": extent,
        "offset": offset,
        "parent": parent_name,
        "spaces": spaces,
        "handles": handles,
        "scale": scale,
        "info": info,
    }


def load_joint_from_config(config, scale=1.0):
    joint_name = config["name"]
    joint_type = config["type"].split("_")[-1]
    tf2parent_6d = np.array(config["tf2parent"])
    tf2parent = np.eye(4)
    tf2parent[:3, 3] = tf2parent_6d[:3] * scale
    tf2parent[:3, :3] = R.from_rotvec(tf2parent_6d[3:]).as_matrix()
    axis_local = np.array(config["axis_local"])
    parent_name = config["parent"]
    limits = np.array(config.get("limits", [0, 1]))
    if joint_type == "prismatic":
        limits = limits * scale  # Apply scale to limits
    child_name = config.get("child", None)
    info = config.get("info", None)
    return {
        "name": joint_name,
        "joint_type": joint_type,
        "tf2parent": tf2parent,
        "axis_local": axis_local,
        "parent": parent_name,
        "limits": limits,
        "child": child_name,
        "scale": scale,
        "info": info,
    }


def load_art_object_from_config(config, scale=1.0, is_robot=False):
    obj_name = config["name"]
    parts = config["parts"]
    info = config.get("info", None)
    assert len(parts) > 0, "No parts in the object."
    # Load base part
    if not is_robot:
        assert parts[0]["name"] == "base" or parts[0]["name"] == "panda_link0", "Base part must be placed at the first position."
    base_extent = np.array(parts[0]["extent"]) * scale
    base_tf_6d = np.array(parts[0].get("tf", [0, 0, 0, 0, 0, 0]))
    base_tf = np.eye(4)
    base_tf[:3, 3] = base_tf_6d[:3] * scale
    base_tf[:3, :3] = R.from_rotvec(base_tf_6d[3:]).as_matrix()
    if "spaces" in parts[0]:
        base_space = Space(**load_space_from_config(parts[0]["spaces"], scale=scale))
    else:
        base_space = None
    if not is_robot:
        art_obj = ArtObject(
            name=parts[0]["name"],
            extent=base_extent,
            tf=base_tf,
            spaces=base_space,
            scale=scale,
            info=info,
        )
    else:
        art_obj = Robot(
            name=parts[0]["name"],
            extent=base_extent,
            tf=base_tf,
            spaces=base_space,
            scale=scale,
            info=info,
        )
    # Load other parts
    for part in parts[1:]:
        if part["type"].startswith("Link"):
            art_obj.add_link(**load_link_from_config(part, scale=scale))
        elif part["type"].startswith("Joint"):
            art_obj.add_joint(**load_joint_from_config(part, scale=scale))
    art_obj.update_level()
    art_obj.forward_pose()
    art_obj.obj_name = obj_name
    return art_obj


if __name__ == "__main__":

    ################################## Test case: Rotate Door ##################################
    space = Space(extent=np.array([0.8, 0.8, 0.8]))
    storage = ArtObject(name="base", extent=np.array([1, 1, 1]), spaces=[space])
    # Door Joint
    tf2parent = np.eye(4)
    tf2parent[:3, 3] = np.array([0.5, 0.5, 0])
    axis_local = np.array([0, 0, -1])
    storage.add_joint("door_joint", "revolute", storage.links["base"], tf2parent, axis_local, None)

    # Door
    tf2parent = np.eye(4)
    tf2parent[:3, 3] = np.array([-0.5, 0.1, 0])
    storage.add_link("door", tf2parent, np.array([1.0, 0.2, 1.0]), storage.joints["door_joint"])

    storage.set_joint_value("door_joint", np.pi / 6)
    storage_vis_o3d = storage.get_vis_o3d()

    ### Add apple
    apple_tf = np.eye(4)
    apple_tf[:3, 3] = np.array([0.3, 0.4, 0.5])
    apple = Link(name="apple", tf=apple_tf, extent=np.array([0.1, 0.1, 0.1]))
    apple_vis_o3d = apple.get_vis_o3d()

    # Visualize
    import open3d as o3d

    orgin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries(storage_vis_o3d + apple_vis_o3d + [orgin])

    # Check collision
    storage_bboxes = storage.get_bboxes()
    apple_bbox = apple.get_bbox()
    collision_statues = check_collision_SAT(apple_bbox, storage_bboxes)
    print(collision_statues)
    ################################## Test case: Slide Drawer ##################################

    # storage = ArtObject()
    # # Base
    # storage.add_link("base", extent=np.array([1, 1, 1]))

    # # Door Joint
    # tf2parent = np.eye(4)
    # tf2parent[:3, 3] = np.array([0.5, 0.5, 0])
    # axis_local = np.array([0, 1, 0])
    # storage.add_joint(
    #     "drawer_joint", "prismatic", storage.links["base"], tf2parent, axis_local, None
    # )

    # # Door
    # tf2parent = np.eye(4)
    # tf2parent[:3, 3] = np.array([-0.5, 0.1, 0])
    # storage.add_link(
    #     "door", tf2parent, np.array([1.0, 1.0, 0.2]), storage.joints["drawer_joint"]
    # )

    # storage.set_joint_value("drawer_joint", 0.3)
    # storage.visualize()
