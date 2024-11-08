from __future__ import annotations
import os
import cv2
import yaml
import trimesh
from urchin import URDF
import json
import shutil
import subprocess
import tempfile
from natsort import natsorted
import mediapy as media
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

try:
    from lgmcts.object_primitives import (
        load_art_object_from_config,
        get_non_overlap_bbox_single,
        get_overlap_bbox_single,
    )
except ImportError:
    from object_primitives import (
        load_art_object_from_config,
        get_non_overlap_bbox_single,
        get_overlap_bbox_single,
    )

#################################### Constants ####################################

DRAWER_NAMES = ["drawer", "handle", "shelf", "other_leaf"]
DOOR_NAMES = ["door", "control_panel", "glass", "mirror", "drawer", "other_leaf"]
BUTTON_NAMES = ["button", "switch", "knob"]

HANG_CATS = ["Dishwasher", "Oven"]


#################################### Math-utils ####################################
# Utils to convert mat44
def mat44_to_tf6d(mat44):
    tf_6d = np.zeros(6)
    tf_6d[:3] = mat44[:3, 3]
    tf_6d[3:] = R.from_matrix(mat44[:3, :3]).as_rotvec()
    return tf_6d


def tf6d_to_mat44(tf_6d):
    mat44 = np.eye(4)
    mat44[:3, 3] = tf_6d[:3]
    mat44[:3, :3] = R.from_rotvec(tf_6d[3:]).as_matrix()
    return mat44


#################################### Media-utils ####################################
def generate_video_from_images(images: str | list[np.ndarray], output_video, output_fps=25):
    """Generate video from images."""
    if output_fps > 0:
        media.write_video(output_video, images, fps=output_fps)
    else:
        # save as images
        for i, img in enumerate(images):
            media.write_image(output_video.replace(".mp4", f"_{i:05d}.png"), img)


def generate_depth_video_from_images(depth_images: str | list[np.ndarray], output_video, output_fps=25, depth_min_clip=2.0, depth_max_clip=5.0):
    """Generate depth video from depth images.
    depth_min_clip: min clip value from camera. (already clipped)
    depth_max_clip: max clip value from camera. (already clipped)
    """
    if isinstance(depth_images, list):
        depth_images = np.stack(depth_images, axis=0)
    N, H, W, C = depth_images.shape
    # Normalize depth images to 8-bit range (0-255)
    depth_max_mask = depth_images >= depth_max_clip
    depth_min_mask = depth_images <= depth_min_clip
    in_region_mask = np.logical_and(depth_images > depth_min_clip, depth_images < depth_max_clip)
    depth_min = depth_images[in_region_mask].min()
    depth_max = depth_images[in_region_mask].max()
    depth_images_8bit = np.zeros((N, H, W, 1), dtype=np.uint8)
    depth_images_8bit[in_region_mask] = ((depth_images[in_region_mask] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    depth_images_8bit = depth_images_8bit.squeeze(axis=-1)
    # # Release the VideoWriter
    # video_writer.release()
    media.write_video(output_video, depth_images_8bit, fps=output_fps, codec="h264", encoded_format="yuv420p")
    return np.array([depth_min, depth_max])


def extract_space_from_mesh(mesh, thresh=0.1, left_ratio=0.1, space_type="shelf"):
    """Extracting space from mesh."""
    # FIXME: Currently, only support opengl structure, as Partnet-mobility-v0
    # Adding bounds, because some objects doesn't upper mesh
    bounds = mesh.bounding_box.bounds
    bounds_z = bounds[1][1] - bounds[0][1]
    bounds_x = bounds[1][0] - bounds[0][0]
    z_thresh = min(thresh, bounds_z * 0.4)  # Dynamic threshold
    x_thresh = min(thresh, bounds_x * 0.4)  # Dynamic threshold
    # # [DEBUG]: visualize parent link & bbox
    # if space_type == "drawer":
    #     scene = trimesh.Scene([mesh])
    #     scene.show()
    if space_type == "drawer":
        # The upper side of drawer is not the whole bbox.
        # Instead, it should be the upper bound of the cutting geometry from the middle.
        # Define the plane: point and normal vector
        plane_origin = (bounds[0] + bounds[1]) / 2  # Middle point of the bounding box
        plane_normal = np.array([0, 0, -1])  # Normal vector of the plane
        cut_mesh = mesh.slice_plane(plane_origin, plane_normal)
        cut_bounds = cut_mesh.bounding_box.bounds
        # scene = trimesh.Scene([cut_mesh])
        # scene.show()
        upper_bound = np.array([0, cut_bounds[1][1], 0])
    else:
        upper_bound = np.array([0, bounds[1][1], 0])

    # Generate arrays to compute intersection
    center = (bounds[0] + bounds[1]) / 2
    z_direction = np.array([0, 1, 0])  # z in opengl structure is y
    ray_origins = [center, center]
    ray_directions = [z_direction, -z_direction]
    z_locs, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)
    if len(z_locs) == 0:
        # Some special structure, open side.
        z_locs = np.array([bounds[0], bounds[1]])
    else:
        z_locs = np.vstack([z_locs, upper_bound])
    z_locs = np.array(sorted(np.dot(z_locs, z_direction)))
    z_gap = z_locs[1:] - z_locs[:-1]
    # Get intersect from x direction
    x_direction = np.array([1, 0, 0])
    ray_origins = [center, center]
    ray_directions = [x_direction, -x_direction]
    x_locs, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)
    if len(x_locs) == 0:
        # Some special structure, open side.
        x_locs = np.array([bounds[0], bounds[1]])
    x_locs = np.array(sorted(np.dot(x_locs, x_direction)))
    x_gap = x_locs[1:] - x_locs[:-1]
    # Filter out those small gaps
    z_space_exists = z_gap > z_thresh
    x_space_exists = x_gap > x_thresh

    space_extents = []
    space_centers = []

    contain_extent = (bounds[1] - bounds[0]).tolist()
    # Compute intersection from z direction for dividing spaces.
    for i in range(0, len(x_space_exists)):
        if x_space_exists[i] == True:
            for j in range(0, len(z_space_exists)):
                if z_space_exists[j] == True:
                    space_extents.append(
                        [
                            x_locs[i + 1] - x_locs[i],  # x
                            z_locs[j + 1] - z_locs[j],  # y
                            (1 - left_ratio) * contain_extent[2],  # z
                        ]
                    )
                    space_center = (
                        np.array(
                            [
                                (x_locs[i + 1] + x_locs[i]) / 2,
                                (z_locs[j + 1] + z_locs[j]) / 2,
                                center[2],
                            ]
                        )
                        - center
                    )
                    space_centers.append(space_center)
    return space_extents, space_centers


def parse_container_information(
    urdf,
    link,
    links,
    link_dict,
    link_names,
    joint_dict,
    completed_search={},
    thresh=0.1,
    **kwargs,
):
    # FIXME: Currently not consider the counter_top space type
    # Leave it for future work
    left_ratio = 0.1
    # Decide the space type
    space_type = "unknown"
    # First, check if it is a drawer
    if any([drawer_name in link["name"].lower() for drawer_name in DRAWER_NAMES]):
        parent_joint = joint_dict[link["raw_parent"]]
        parent_link = link_dict[parent_joint.parent]
        if parent_joint.joint_type != "prismatic":
            # Not a drawer
            space_type = "unknown"
        else:
            space_type = "drawer"
        mesh = link_dict[link["raw_name"]].collision_mesh
        space_name = link["raw_name"]
        if space_name in completed_search:
            return space_name, space_type, link["name"], []
        else:
            completed_search[space_name] = 1
    # Second, check if it is a shelf type
    if space_type == "unknown" and any([door_name in link["name"].lower() for door_name in DOOR_NAMES]):
        # Get the parent link of the door. (Ideally, this is the base.)
        parent_joint = joint_dict[link["raw_parent"]]
        parent_link = link_dict[parent_joint.parent]
        mesh = parent_link.collision_mesh
        space_name = parent_link.name
        if mesh is None:
            return space_name, space_type, link["name"], []

        if space_name in completed_search:
            return space_name, space_type, link["name"], []
        else:
            completed_search[space_name] = 1
        space_type = "shelf"
    elif space_type == "unknown":
        return "", link["name"], space_type, []

    # Extract space from mesh
    # try:
    space_extents, space_centers = extract_space_from_mesh(mesh, thresh, left_ratio, space_type)
    # except Exception as e:
    #     assert False, f"Error: {e}, data_id: {kwargs['data_id']}"
    # Return Spaces.
    link_spaces = []
    for idx, (space_extent, space_center) in enumerate(zip(space_extents, space_centers)):
        tf2parent = np.zeros(6)
        tf2parent[:3] = space_center
        link_spaces.append(
            {
                "name": f"{link_names.get(space_name, space_name)}_{idx}",
                "extent": [float(x) for x in space_extent],
                "tf2parent": tf2parent.tolist(),
                "dir_up": ["+y"],  # For opengl object, it is +y
                "trigger": [],  # Trigger to open the space
                "space_type": space_type,
                "is_visible": False,
                "is_reachable": False,
            }
        )
    return space_name, link["name"], space_type, link_spaces


def create_table_top_space(root_link):
    """Create table top space for the articulated object."""
    space_height = 0.3
    table_top_extent = np.array(root_link["extent"])
    table_top_extent[1] = space_height  # y-axis, as this is opengl orient
    tf2parent = np.eye(4)
    tf2parent[1, 3] = root_link["extent"][1] / 2.0 + table_top_extent[1] / 2.0
    space_type = "open"
    table_top_space = {
        "name": f"{root_link['name']}_table-top",
        "extent": table_top_extent.tolist(),
        "tf2parent": mat44_to_tf6d(tf2parent).tolist(),
        "dir_up": ["+y"],
        "trigger": [["all", 1.0]],  # Trigger to open the space
        "space_type": space_type,
        "is_visible": True,  # Open space
        "is_reachable": True,
    }
    if "spaces" not in root_link:
        root_link["spaces"] = []
    root_link["spaces"].append(table_top_space)


def split_space_by_trigger(art_config, space_dict, filter_thresh=0.2):
    """Currently, we only consider the space that have at most one trigger.
    So what we are going to do here is to fix the space so that it matches the trigger size
    from the direction that the space is supposed to be opened.
    """
    # FIXME: Currently only consider the one layer of abstraction
    art_obj = load_art_object_from_config(art_config, scale=1.0, is_robot=False)

    # Project everything to 2d.
    # [DEBUG]
    def vis_art_obj(art_obj):
        import open3d as o3d

        vis = art_obj.get_vis_o3d(show_joint=True)
        for space in art_obj.spaces.values():
            vis += space.get_vis_o3d()
        o3d.visualization.draw_geometries(vis)

    replace_map = {}
    space_trigger_value = 0.9
    for space_name, space in art_obj.spaces.items():
        contain_link_name = "_".join(space_name.split("_")[:-1])
        contain_link = art_obj.links[contain_link_name]
        space_open_dir = space.adj_m[:3, :3] @ space.opened_dirs[0]
        if np.abs(space_open_dir[1]) > 0.9:
            space_open_dir = "y"
        elif np.abs(space_open_dir[0]) > 0.9:
            space_open_dir = "x"
        else:
            space_open_dir = "z"
        if space.space_type == "drawer":
            # Drawer branch
            joint_idx = art_obj.get_joint_idx(contain_link_name)
            joint_values = np.zeros(len(art_obj.active_joints))
            joint_values[joint_idx] = space_trigger_value
            art_obj.set_joint_values(joint_values, is_rel=True)
            # Get the space bbox
            space_bbox_array = np.array(
                [
                    space.tf[:3, 3],
                    R.from_matrix(space.tf[:3, :3]).as_rotvec(),
                    space.extent,
                ]
            )
            # Compute the not-overlapping space
            whole_bbox_min, whole_bbox_max = art_obj.get_whole_bbox(block_list=[contain_link_name])
            whole_bbox_ext = whole_bbox_max - whole_bbox_min
            whole_bbox_center = (whole_bbox_max + whole_bbox_min) / 2
            whole_bbox_array = np.array(
                [
                    whole_bbox_center,
                    [
                        0,
                        0,
                        0,
                    ],
                    whole_bbox_ext,
                ]
            )
            # Get the nno bbox
            bbox_offset_nno, bbox_extent_nno = get_non_overlap_bbox_single(space_bbox_array, whole_bbox_array, proj_axis=space_open_dir)
            if np.prod(bbox_extent_nno) < 1e-8:
                continue
            else:
                # Update the space
                space.extent = bbox_extent_nno
                space.tf2parent[:3, 3] += bbox_offset_nno
            # vis_art_obj(art_obj)
            # For drawer object, the trigger is the drawer.
            space.trigger = [[contain_link_name, space_trigger_value]]
            # Save the replace_map
            replace_map[space_name] = [space]
        elif space.space_type == "shelf":
            # Process shelf cases. Shelf structure is more complex.
            # Because for shelf space, it can have more than one trigger.
            # So we need to split the space into multiple spaces.
            # First, prepare the possible trigger list. Shelf is paired with doors.
            # Get the space bbox
            joint_values = np.zeros(len(art_obj.active_joints))
            art_obj.set_joint_values(joint_values, is_rel=True)
            # vis_art_obj(art_obj)
            space_bbox_array = np.array(
                [
                    space.tf[:3, 3],
                    R.from_matrix(space.tf[:3, :3]).as_rotvec(),
                    space.extent,
                ]
            )
            space_copy_list = []
            for child_joint in contain_link.children:
                child_link_name = child_joint.child.name
                if any([door_name in child_link_name.lower() for door_name in DRAWER_NAMES]):
                    if child_joint.joint_type == "prismatic":
                        continue  # Skip the drawer
                if any([door_name in child_link_name.lower() for door_name in DOOR_NAMES]):
                    door_link = child_joint.child
                    door_bbox = np.array(
                        [
                            door_link.tf[:3, 3],
                            R.from_matrix(door_link.tf[:3, :3]).as_rotvec(),
                            door_link.extent,
                        ]
                    )
                    bbox_offset_over, bbox_extent_over = get_overlap_bbox_single(space_bbox_array, door_bbox, proj_axis=space_open_dir)
                    if np.prod(bbox_extent_over) < 1e-8 or any(bbox_extent_over / (space.extent + 1e-8) < filter_thresh):
                        continue
                    else:
                        # Get a space_copy
                        space_copy = deepcopy(space)
                        space_copy.name = f"{space_name}_{child_link_name}"
                        space_copy.extent = bbox_extent_over
                        space_copy.tf2parent[:3, 3] += bbox_offset_over
                        space_copy.trigger = [[child_link_name, space_trigger_value]]
                        space_copy_list.append(space_copy)
                # UPdate the replace_map
                replace_map[space_name] = space_copy_list
        elif space.space_type == "open":
            replace_map[space_name] = [space]
        else:
            raise ValueError(f"Unknown space type: {space.space_type}")
    # Update art_config
    space_count = 0
    for obj in art_config["parts"]:
        if obj["type"] == "Link":
            if "spaces" in obj:
                new_spaces = []
                for space in obj["spaces"]:
                    space_name = space["name"]
                    if space_name in replace_map:
                        for new_space in replace_map[space_name]:
                            new_spaces.append(
                                {
                                    "name": new_space.name,
                                    "extent": new_space.extent.tolist(),
                                    "tf2parent": mat44_to_tf6d(new_space.tf2parent).tolist(),
                                    "dir_up": ["+y"],
                                    "trigger": new_space.trigger,
                                    "space_type": new_space.space_type,
                                    "is_visible": new_space.is_visible,
                                    "is_reachable": new_space.is_reachable,
                                }
                            )
                            space_count += 1
                obj["spaces"] = new_spaces
    return art_config, space_count


def parse_handle_information(link, links, link_dict):
    if any([door_name in link["name"].lower() for door_name in DOOR_NAMES]) or any(
        [drawer_name in link["name"].lower() for drawer_name in DRAWER_NAMES]
    ):
        # Get the parent link of the door. (Ideally, this is the base.)
        link_urdf = link_dict[link["raw_name"]]
        # link_visual_names = [v.name for v in link_urdf.visuals]
        handle_dict_list = []
        # Get link mesh
        link_mesh_list = []
        handle_mesh_list = []
        # for visual_obj in link_urdf.visuals:
        #     obj_origin = visual_obj.origin
        #     obj_meshes = visual_obj.geometry.mesh.meshes
        #     link_mesh_list.extend([m.apply_transform(obj_origin) for m in obj_meshes])
        # link_mesh = trimesh.util.concatenate(link_mesh_list)
        # link_bbox = link_mesh.bounding_box.bounds
        # link_center = (link_bbox[0] + link_bbox[1]) / 2.0
        # Get handle mesh
        for visual_obj in link_urdf.visuals:
            obj_origin = visual_obj.origin
            obj_meshes = visual_obj.geometry.mesh.meshes
            obj_meshes = [m.apply_transform(obj_origin) for m in obj_meshes]
            link_mesh_list.extend(obj_meshes)
            if "handle" in visual_obj.name.lower():
                # Handle meshes
                handle_mesh = trimesh.util.concatenate(obj_meshes)
                handle_mesh_list.append(handle_mesh)
        # Compute the handle bbox & offset
        link_mesh = trimesh.util.concatenate(link_mesh_list)
        link_bbox = link_mesh.bounding_box.bounds
        link_center = (link_bbox[0] + link_bbox[1]) / 2.0
        for handle_mesh in handle_mesh_list:
            # Handle mesh
            handle_bbox = handle_mesh.bounding_box.bounds  # Z-out is the handle outside dir.
            # Export it as a dict
            handle_center = (handle_bbox[0] + handle_bbox[1]) / 2.0
            tf2parent = np.zeros(6)
            tf2parent[:3] = handle_center - link_center
            handle_extent = handle_bbox[1] - handle_bbox[0]
            handle_dict = {
                "extent": [float(x) for x in handle_extent],
                "tf2parent": tf2parent.tolist(),
                "dir_up": ["+y"],  # For opengl object, it is +y
            }
            handle_dict_list.append(handle_dict)
        # # [DEBUG]
        # scene = trimesh.Scene()
        # for handle_mesh in handle_mesh_list:
        #     scene.add_geometry(handle_mesh)
        # scene.show()
        return handle_dict_list
    else:
        return []


# Function to build YAML content as a string
def build_yaml_content(urdf, mobility_info={}, model_cat="", data_id=0, verbose=False, info={}):
    # Part mobility info
    joint_names = {"base": "base"}
    link_names = {"base": "base"}
    for joint_info in mobility_info:
        joint_id = joint_info["id"]
        joint_names[f"joint_{joint_id}"] = f"{joint_info['name']}_{joint_id}"
        link_names[f"link_{joint_id}"] = f"{joint_info['name']}_{joint_id}"
        # FIXME: Why do I need part?
        # for part_info in joint_info["parts"]:
        #     part_id = part_info["id"]
        #     link_names[f"link_{part_id}"] = f"{part_info['name']}_{part_id}"

    output_dict = {
        "name": f"{model_cat}_{data_id}",
        "type": "ArtObject",
        "parts": [],
        "info": info,
    }
    # Storage structure
    link_dict = {}
    joint_dict = {}
    links = []
    link_name2idx = {}
    # Adding links
    for link in urdf.links:
        link_info = {}
        if link.name.endswith("helper"):
            continue
        link_info["name"] = link_names.get(link.name, link.name)
        link_info["raw_name"] = link.name
        link_info["info"] = {"raw_name": link.name}
        if link.name != "base":
            parent = ""
            for joint in urdf.joints:
                if joint.child == link.name:
                    parent = joint.name
                    break
            link_info["parent"] = joint_names.get(parent, parent)
            link_info["raw_parent"] = parent
        link_info["type"] = "Link"
        if link.name == "base":
            extent = [0.01, 0.01, 0.01]
            link_info["extent"] = extent
            tf = [0, 0, 0, 0, 0, 0]
            link_info["tf"] = tf
        else:
            extent = [0.1, 0.1, 0.1]
            if link.collision_mesh:
                bounds = link.collision_mesh.bounding_box.bounds
                extent = (bounds[1] - bounds[0]).tolist()
                offset = (bounds[1] + bounds[0]) / 2
            else:
                # raise ValueError("No collision mesh found")
                extent = [0.1, 0.1, 0.1]
                offset = [0, 0, 0]
                print(f"[Warning]: No collision mesh found for {link.name}")
            link_info["extent"] = extent
            tf = [float(offset[0]), float(offset[1]), float(offset[2]), 0, 0, 0]
            link_info["tf2parent"] = tf
        if verbose:
            print(f"Name: {link_names.get(link.name, link.name)}, extent: {extent}")
        links.append(link_info)
        output_dict["parts"].append(link_info)
        link_dict[link.name] = link
        link_name2idx[link_info["name"]] = len(links) - 1

    # Adding joints
    for joint in urdf.joints:
        joint_info = {}
        if joint.name.endswith("helper"):
            # FIXME: currently, don't consider helper joint
            continue
        # Joint type process
        joint_type = joint.joint_type
        if joint_type == "continuous":
            joint_type = "revolute"
        joint_origin = joint.origin
        joint_axis = joint.axis.tolist()

        # Write
        joint_info["name"] = joint_names.get(joint.name, joint.name)
        joint_info["raw_name"] = joint.name
        joint_info["type"] = f"Joint_{joint_type}"
        joint_info["parent"] = link_names.get(joint.parent, joint.parent)
        joint_info["raw_parent"] = joint.parent
        joint_info["child"] = link_names.get(joint.child, joint.child)
        joint_info["info"] = {"raw_name": joint.name}
        if joint_axis is not None:
            joint_info["axis_local"] = joint_axis
        else:
            print("Error: No joint axis, yaml will not function")
        # Compute the rel_T
        if joint_origin is not None:
            tf6d = mat44_to_tf6d(joint_origin).tolist()
            joint_info["tf2parent"] = tf6d
        else:
            print("Error: No joint tf2parent, yaml will not function")

        if joint.limit is not None:
            limits = [float(joint.limit.lower), float(joint.limit.upper)]
        else:
            limits = [-np.pi, np.pi]
        joint_info["limits"] = limits
        output_dict["parts"].append(joint_info)
        joint_dict[joint.name] = joint

    # Adding space
    completed_search = {}

    # TODO: Add space information here
    space_dict = {}
    for link in links:
        if "raw_parent" not in link:
            # Root link
            continue
        space_link_name, space_type, link_name, link_spaces = parse_container_information(
            urdf,
            link,
            links,
            link_dict,
            link_names,
            joint_dict,
            completed_search,
            data_id=data_id,
        )
        if link_spaces:
            space_link_idx = link_name2idx[link_names.get(space_link_name, space_link_name)]
            if link_spaces and ("spaces" not in links[space_link_idx]):
                links[space_link_idx]["spaces"] = link_spaces
                for space in link_spaces:
                    space_dict[space["name"]] = space
        # Adding handle information
        handles = parse_handle_information(link, links, link_dict)
        if handles:
            link["handles"] = handles
    # Add table top space
    ## Locate root link
    for joint in joint_dict.values():
        if joint.parent == "base" or joint.parent == "panda_link0":
            root_link_name = joint.child
            break
    for link in links:
        if link["raw_name"] == root_link_name:
            root_link = link
            break
    create_table_top_space(root_link)
    # Add triggers for spaces & perform filter
    output_dict, space_count = split_space_by_trigger(output_dict, space_dict)
    output_dict["info"]["space_count"] = space_count
    return yaml.dump(output_dict)


def create_yaml_from_partnet(
    partnet_dir,
    output_dir,
    data_id,
    urdf_file="mobility.urdf",
    verbose=False,
    jump_empty=False,
):
    # Load the URDF file
    part_file = os.path.join(partnet_dir, urdf_file)
    try:
        robot = URDF.load(part_file)
    except Exception as e:
        print(f"Load {partnet_dir} failed... with {e}")
        return False

    # [DEBUG]
    # robot.show()
    # Load meta file
    meta_file = os.path.join(partnet_dir, "meta.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    model_cat = meta["model_cat"]

    json_file = os.path.join(partnet_dir, "mobility_v2.json")
    with open(json_file, "r") as f:
        mobility_info = json.load(f)
    yaml_content = build_yaml_content(
        robot,
        mobility_info,
        model_cat,
        data_id,
        verbose=verbose,
        info={"urdf_file": os.path.join(os.path.basename(os.path.dirname(part_file)), urdf_file)},
    )

    if jump_empty and yaml_content["info"]["space_count"] == 0:
        print(f"Skip {data_id} due to there exists no space...")
        return False
    # Write YAML content line by line to the file
    output_file = os.path.join(output_dir, f"{data_id}.yaml")
    with open(output_file, "w") as yaml_file:
        for line in yaml_content.splitlines(True):
            yaml_file.write(line)
    return True


def create_yaml_for_robot(robot_desc_dir, robot_urdf_file, output_dir, ee_name="panda_hand"):
    # Replacing the current urdf file content by replacing
    # package:// with the robot_desc_dir
    robot_name = robot_urdf_file.split("/")[-1].split(".")[0]
    with open(robot_urdf_file, "r") as f:
        content = f.read()
    content = content.replace("package://", f"{robot_desc_dir}/")
    robot_urdf_file_local = os.path.join(robot_desc_dir, f"{robot_name}.urdf")
    with open(robot_urdf_file_local, "w") as f:
        f.write(content)

    robot = URDF.load(robot_urdf_file_local)
    yaml_content = build_yaml_content(
        robot,
        model_cat=robot_name,
        info={"urdf_file": robot_urdf_file_local, "ee_name": ee_name},
    )

    # Write YAML content line by line to the file
    output_file = os.path.join(output_dir, f"{robot_name}.yaml")
    with open(output_file, "w") as yaml_file:
        for line in yaml_content.splitlines(True):
            yaml_file.write(line)
    return True


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    franka_desc_dir = os.path.join(cur_dir, "assets/robo_urdf")

    # # Robot
    # franka_file = os.path.join(franka_desc_dir, "franka_description/robots", "franka_panda.urdf")
    franka_file = os.path.join(franka_desc_dir, "panda_newgripper.urdf")
    output_dir = "assets/robot_templates"
    os.makedirs(output_dir, exist_ok=True)
    create_yaml_for_robot(franka_desc_dir, franka_file, output_dir)

    # partnet_id = "45219"
    # create_yaml_from_partnet(
    #     f"{cur_dir}/raw_data/partnet-mobility-v0/dataset/{partnet_id}",
    #     f"{cur_dir}/config/art_templates",
    #     partnet_id,
    #     verbose=True,
    # )
