"""Process partnet datas."""

import numpy as np
from tqdm import tqdm
import trimesh
import os
import yaml
import json
from copy import deepcopy
import open3d as o3d
from urchin import URDF
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from lgmcts.utils import create_yaml_from_partnet
from lgmcts.object_primitives import (
    load_art_object_from_config,
    ArtObject,
    Link,
    Joint,
    Handle,
    check_collision_SAT,
)
from lgmcts.geometry_utils import create_textured_cuboid, resize_textured_obj

###################### Constants ###################################
HANDLE_MIN_DEPTH = 0.12
HANDLE_MAX_WIDTH = 0.04
HANDLE_WH_RATIO = 5.0

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
HANDLE_TEMPLATES = [f"{CUR_DIR}/resources/handles/type_1/original-45.obj"]


###################### Utils ###################################
def replace_obj_files_in_urdf(urdf_root, target_obj, visual_obj, collision_obj):
    for link in urdf_root.findall(".//link"):
        for visual in link.findall(".//visual"):
            mesh = visual.find(".//mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                if filename and target_obj in filename:
                    if visual_obj:
                        mesh.set("filename", visual_obj)  # Replace the file for visual tag
                    else:
                        link.remove(visual)
        for collision in link.findall(".//collision"):
            mesh = collision.find(".//mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                if filename and target_obj in filename:
                    if collision_obj:
                        mesh.set("filename", collision_obj)  # Replace the file for collision tag
                    else:
                        link.remove(collision)


def add_meshes_to_link(urdf_file, link_names, mesh_names, mesh_files, texture_files, pose_xyzs, pose_rpys):
    """Add a mesh to a link in the URDF file."""
    # Parse the URDF file
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Define the namespace if needed
    # ns = {"": "http://www.ros.org"}

    for link_name, mesh_name, mesh_file, texture_file, pose_xyz, pose_rpy in zip(
        link_names, mesh_names, mesh_files, texture_files, pose_xyzs, pose_rpys
    ):
        # Find the link by name
        link = root.find(f".//link[@name='{link_name}']")

        if link is None:
            print(f"Link '{link_name}' not found in the URDF.")
            return

        # Create the <visual> tag
        visual = ET.SubElement(link, "visual")
        visual.set("name", mesh_name)

        # Create the <origin> tag
        origin = ET.SubElement(visual, "origin")
        origin.set("xyz", " ".join(map(str, pose_xyz)))
        origin.set("rpy", " ".join(map(str, pose_rpy)))

        # Create the <geometry> tag
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        mesh.set("filename", mesh_file)

        # # Optional: Add texture
        # if texture_files is not None:
        #     material = ET.SubElement(visual, "material")
        #     texture = ET.SubElement(material, "texture")
        #     texture.set("filename", texture_file)

        # Optionally add the same mesh to the collision (comment out if not needed)
        collision = ET.SubElement(link, "collision")
        origin_collision = ET.SubElement(collision, "origin")
        origin_collision.set("xyz", " ".join(map(str, pose_xyz)))
        origin_collision.set("rpy", " ".join(map(str, pose_rpy)))
        geometry_collision = ET.SubElement(collision, "geometry")
        mesh_collision = ET.SubElement(geometry_collision, "mesh")
        mesh_collision.set("filename", mesh_file)

    # Write back to the URDF file (or save to a new file)
    modified_urdf_file = urdf_file.replace("_v1.urdf", "_v2.urdf")
    tree.write(modified_urdf_file, xml_declaration=True, encoding="utf-8", method="xml")
    print(f"URDF file has been modified and saved as {modified_urdf_file}.")


###################### Functions ###################################


def create_table_top(art_obj: ArtObject, base_link: Link):
    """Some partnet object doesn't have table surface, we need to add it manually."""
    absolute_height = 0.01
    table_top_extent = np.copy(base_link.extent)
    table_top_extent[1] = absolute_height  # y-axis, as this is opengl orient
    tf2parent = np.eye(4)
    tf2parent[1, 3] = base_link.extent[1] / 2.0 + table_top_extent[1] / 2.0
    table_top = Link(
        name="table_top",
        extent=table_top_extent,
        tf2parent=tf2parent,
        tf=base_link.tf @ tf2parent,
    )
    return table_top


def replace_handle(export_dir, urdf, raw_urdf_file):
    """Replace the handle collision obj in urdf file with a bbox obj. This is to make the objects easier to grasp."""
    replaced_results = []
    for link in urdf.links:
        handle_objs = [v for v in link.visuals if "handle" in v.name.lower()]
        handle_mesh_lists = [v.geometry.mesh.meshes for v in handle_objs]
        handle_mesh_files = [v.geometry.mesh.filename for v in handle_objs]
        if len(handle_mesh_lists) == 0:
            continue
        else:
            handle_meshes = sum([v for v in handle_mesh_lists], [])
            handle_mesh_whole = trimesh.util.concatenate(handle_meshes)
            handle_bounds = handle_mesh_whole.bounds
            handle_center = handle_mesh_whole.centroid
            # link bounds
            link_mesh_lists = [v.geometry.mesh.meshes for v in link.visuals]
            link_meshes = sum([v for v in link_mesh_lists], [])
            link_mesh_whole = trimesh.util.concatenate(link_meshes)
            link_bounds = link_mesh_whole.bounds
            link_center = link_mesh_whole.centroid

            # # [DEBUG]
            # scene = trimesh.scene.scene.Scene(geometry=[handle_mesh_whole, link_mesh_whole])
            # scene.show()
            # For the scale_xy, there are two constraints:
            # 1. We want to make the handle width to be HANDLE_MAX_WIDTH if possible
            # 2. We don't want to make the handle's bounds exceed the link's bounds
            bound_region_xy = np.abs(link_bounds[1, :2] - link_bounds[0, :2]) / 2.0 - np.abs(handle_center[:2] - link_center[:2])
            _scale_xy_max = np.clip(2 * bound_region_xy[:2] / (handle_bounds[1, :2] - handle_bounds[0, :2] + 1e-6), a_min=1.0, a_max=None)
            # width is the shorted edge
            handle_extent = handle_bounds[1, :2] - handle_bounds[0, :2]
            link_extent = link_bounds[1, :2] - link_bounds[0, :2]
            long_axis = np.argmax(handle_extent)
            if np.min(handle_extent) < 1e-6:
                ill_handle = True
            if long_axis == 0:
                # x-axis long
                _scale_y = min(HANDLE_MAX_WIDTH / (handle_extent[1] + 1e-6), _scale_xy_max[1])
                _scale_x = min(_scale_y, _scale_xy_max[0])
            else:
                # y-axis long
                _scale_x = min(HANDLE_MAX_WIDTH / (handle_extent[0] + 1e-6), _scale_xy_max[0])
                _scale_y = min(_scale_x, _scale_xy_max[1])
            _scale_x, _scale_y = min(_scale_x, _scale_y), min(_scale_x, _scale_y)
            # max Z-value after scaling
            _scale_z = max(HANDLE_MIN_DEPTH / (handle_bounds[1, 2] - handle_bounds[0, 2] + 1e-6), 1.0)
            max_z = (handle_bounds[1, 2] - handle_bounds[0, 2]) * _scale_z + handle_bounds[0, 2]
            z_expand = (handle_bounds[1, 2] - handle_bounds[0, 2]) * _scale_z
            # compute rescale offset
            rescale = np.array([_scale_x, _scale_y, _scale_z])
            rescale_offset = handle_center * (1 - rescale)
            if handle_center[2] > link_center[2]:
                rescale_offset[2] = handle_bounds[0, 2] * (1 - _scale_z)
            else:
                # Inverse direction
                rescale_offset[2] = handle_bounds[1, 2] * (1 - _scale_z)
            # # [DEBUG]
            # scene = trimesh.scene.scene.Scene(geometry=[handle_mesh_whole])
            # scene.show()
            for handle_mesh_list, handle_mesh_file in zip(handle_mesh_lists, handle_mesh_files):
                handle_mesh = trimesh.util.concatenate(handle_mesh_list)
                bbox = handle_mesh.bounding_box
                # Save the bbox mesh to replace the handle mesh
                # Make the handle twice thicker
                bbox_extents = np.copy(bbox.extents)
                bbox_extents[0] *= _scale_x
                bbox_extents[1] *= _scale_y
                bbox_extents[2] *= z_expand / (bbox_extents[2] + 1e-6)

                ##################### Use bbox as mesh ########################
                bbox_mesh = trimesh.creation.box(extents=bbox_extents)
                bbox_mesh.apply_transform(bbox.primitive.transform)
                # z_offset = max_z - bbox_mesh.bounds[1, 2]
                # bbox_mesh.apply_translation([0, 0, z_offset])

                # Scale the visual mesh along z-axis
                handle_mesh.apply_scale(rescale)
                handle_mesh.apply_translation(rescale_offset)
                # align the center of handle_mesh & bbox_mesh
                bbox_mesh.apply_translation(handle_mesh.centroid - bbox_mesh.centroid)

                handle_mesh_name = handle_mesh_file.split("/")[-1].split(".")[0]
                os.makedirs(os.path.join(export_dir, "textured_objs", f"{handle_mesh_name}_vis"), exist_ok=True)
                handle_vis_file = os.path.join("textured_objs", f"{handle_mesh_name}_vis", f"{handle_mesh_name}_visual.obj")
                # rename texture if exists
                if hasattr(handle_mesh, "visual") and hasattr(handle_mesh.visual, "material"):
                    # Access the material
                    material = handle_mesh.visual.material
                    # Rename the material (assuming the material has a 'name' attribute)
                    material.name = f"{handle_mesh_name}_vis"
                handle_mesh.export(os.path.join(export_dir, handle_vis_file))

                # Put on silver color
                bbox_mesh.visual.face_colors = [200, 200, 200, 255]
                os.makedirs(os.path.join(export_dir, "textured_objs", f"{handle_mesh_name}_collision"), exist_ok=True)
                handle_collision_file = os.path.join("textured_objs", f"{handle_mesh_name}_collision", f"{handle_mesh_name}_collision.obj")
                bbox_mesh.export(os.path.join(export_dir, handle_collision_file))
                if not ill_handle:
                    replaced_results.append([handle_mesh_file, handle_vis_file, handle_collision_file])
                else:
                    replaced_results.append([handle_mesh_file, None, None])

                ###################### Load existing mesh ########################
                # # [DEBUG]
                # scene = trimesh.scene.scene.Scene(geometry=[handle_mesh, link_mesh_whole])
                # scene.show()

    # Replace the handle mesh in the URDF file
    urdf_tree = ET.parse(raw_urdf_file)
    urdf_root = urdf_tree.getroot()
    for target_obj, visual_obj, collision_obj in replaced_results:
        # print(f"Replacing {target_obj} with {visual_obj} and {collision_obj}.")
        replace_obj_files_in_urdf(urdf_root, target_obj, visual_obj, collision_obj)
    # Export
    export_urdf_file = os.path.join(export_dir, raw_urdf_file.split("/")[-1].replace(".urdf", "_v1.urdf"))
    urdf_tree.write(export_urdf_file, xml_declaration=True, encoding="utf-8", method="xml")
    # print(f"URDF file has been modified and saved as {export_urdf_file}.")


def create_handle(art_obj: ArtObject, link: Link, joint: Joint):
    """Create a handle for the given link and joint."""
    joint_type = joint.joint_type
    handle_ratio = 0.3
    margin_ratio = 0.1
    if joint_type == "prismatic":
        # Prismatic joint, meaning the link is a drawer-like object
        # But is also possible, this is a sliding door
        # For prismatic joint, the handle should be placed at the center of the z-axis
        joint_axis_local = joint.axis_local
        if np.abs(joint_axis_local[2]) > 0.5:
            # Drawer-like object
            long_axis = np.argmax(link.extent[:2])
            if long_axis == 0:  # x-axis long
                handle_extent = [HANDLE_WH_RATIO * HANDLE_MAX_WIDTH, HANDLE_MAX_WIDTH, HANDLE_MIN_DEPTH]
            else:  # y-axis long
                handle_extent = [HANDLE_MAX_WIDTH, HANDLE_WH_RATIO * HANDLE_MAX_WIDTH, HANDLE_MIN_DEPTH]
            handle_extent[2] = min(HANDLE_MIN_DEPTH, handle_ratio / 2.0 * link.extent[2])  # z-axis
        else:
            # Sliding door
            handle_extent = [HANDLE_MAX_WIDTH, HANDLE_WH_RATIO * HANDLE_MAX_WIDTH, HANDLE_MIN_DEPTH]
        # CLIP size
        handle_extent[:2] = np.clip(handle_extent[:2], 0.0, link.extent[:2])
        # The bottom-z of handle should be at the top z of the link
        tf2parent = np.eye(4)
        tf2parent[2, 3] = link.extent[2] / 2.0
        handle = Handle(
            name=f"{link.name}_handle",
            extent=handle_extent,
            tf2parent=tf2parent,
            tf=link.tf @ tf2parent,
        )
    elif joint_type == "revolute":
        # Revolute joint, meaning the link is a door-like object
        # For prismatic joint, the handle should be placed at the center of the z-axis
        handle_extent = handle_ratio * link.extent
        handle_extent[0] = min(HANDLE_MAX_WIDTH, handle_ratio * link.extent[0])  # x-axis
        handle_extent[1] = min(HANDLE_MAX_WIDTH, handle_ratio * link.extent[1])  # y-axis
        handle_extent[2] = min(HANDLE_MIN_DEPTH + link.extent[2], 0.1)  # z-axis
        # The bottom-z of handle should be at the bottom z of the link
        tf2parent = np.eye(4)
        tf2parent[2, 3] = -link.extent[2] / 2.0 + handle_extent[2] / 2.0
        # Align the handle to a side of the door
        # Side depends on the axis of the revolute joint
        link_tf2parent = link.tf2parent
        joint_axis_local = joint.axis_local
        if np.abs(joint_axis_local[0]) > np.abs(joint_axis_local[1]):
            # joint is along x-axis, align the handle to the opposite y-axis
            margin = link.extent[1] * margin_ratio / 2.0
            tf2parent[1, 3] = np.sign(link_tf2parent[1, 3]) * (link.extent[1] / 2.0 - handle_extent[1] / 2.0 - margin)
            # Extend the handle along x-axis
            handle_extent[0] = min(handle_extent[0] * HANDLE_WH_RATIO, link.extent[0])
        else:
            # joint is along y-axis, align the handle to the opposite x-axis
            margin = link.extent[1] * margin_ratio / 2.0
            tf2parent[0, 3] = np.sign(link_tf2parent[0, 3]) * (link.extent[0] / 2.0 - handle_extent[0] / 2.0 - margin)
            # Extend the handle along y-axis
            handle_extent[1] = min(handle_extent[1] * HANDLE_WH_RATIO, link.extent[1])
        handle = Handle(
            name=f"{link.name}_handle",
            extent=handle_extent,
            tf2parent=tf2parent,
            tf=link.tf @ tf2parent,
        )
    else:
        raise NotImplementedError(f"Joint type {joint_type} is not supported.")
    # # Check if handle has collision with the rest part
    # other_part_bboxes = art_obj.get_bboxes(block_list=[link.name])
    # collision_status = check_collision_SAT(handle.get_bbox(), other_part_bboxes)
    # if any(collision_status):
    #     print(f"Collision detected for {link.name}'s handle.")
    #     handle = None
    # else:
    #     link.add_handle(handle)
    link.add_handle(handle)
    return handle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--partnet_data_dir", type=str, default="./raw_data/dataset")
    parser.add_argument("--assets_dir", type=str, default="./assets")
    parser.add_argument("--urdf_export_dir", type=str, default="art_urdf")
    parser.add_argument("--texture_dir", type=str, default="./resources/textures")
    parser.add_argument("--template_export_dir", type=str, default="art_templates")
    parser.add_argument("--urdf_name", type=str, default="mobility.urdf")
    parser.add_argument("--cats_compute", type=str, default="StorageFurniture,Microwave,Table,Dishwasher,Oven")
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    assets_dir = args.assets_dir
    partnet_data_dir = args.partnet_data_dir
    urdf_export_dir = os.path.join(assets_dir, args.urdf_export_dir)
    template_export_dir = os.path.join(assets_dir, args.template_export_dir)
    texture_dir = args.texture_dir
    urdf_name = args.urdf_name
    id = args.id

    cats_compute = args.cats_compute.split(",")
    debug = args.debug
    os.makedirs(urdf_export_dir, exist_ok=True)
    os.makedirs(template_export_dir, exist_ok=True)
    data_ids = os.listdir(partnet_data_dir)
    if id is None:
        data_ids = [data_id for data_id in data_ids if data_id.isdigit()]
    else:
        data_ids = [str(id)]
    # data_ids = ["7290"]  # For debug
    rng = np.random.RandomState(0)
    textures = os.listdir(texture_dir)
    handle_texture = [v for v in textures if "handle" in v.lower()]
    handle_texture = [os.path.join(texture_dir, v) for v in handle_texture]
    table_top_texture = [v for v in textures if "table" in v.lower()]
    table_top_texture = [os.path.join(texture_dir, v) for v in table_top_texture]

    ##################### Compute the a selected models #####################
    ids_compute = []
    storage_ids = []  # Some storage furniture doesn't have table top
    for data_id in data_ids:
        meta_file = os.path.join(partnet_data_dir, data_id, "meta.json")
        with open(meta_file, "r") as f:
            meta = json.load(f)
        model_cat = meta["model_cat"]
        if model_cat in cats_compute:
            ids_compute.append(data_id)
        if model_cat == "StorageFurniture":
            storage_ids.append(data_id)

    valid_ids = []
    for data_id in tqdm(ids_compute):
        data_dir = os.path.join(partnet_data_dir, data_id)
        # Try loading
        raw_urdf_file = os.path.join(data_dir, urdf_name)
        try:
            urdf = URDF.load(raw_urdf_file)
        except Exception as e:
            print(f"Failed to load {raw_urdf_file}, error: {e}")
            continue
        valid_ids.append(data_id)
        # Copy the folder to export dir
        export_data_dir = os.path.join(urdf_export_dir, data_id)
        os.makedirs(export_data_dir, exist_ok=True)
        os.system(f"cp -r {data_dir}/* {export_data_dir}")
        # Some handles is too small to grasp, we need to enlarge them
        replace_handle(export_data_dir, urdf, raw_urdf_file)
        # Copy the meta info to export dir
        meta_file = os.path.join(partnet_data_dir, data_id, "meta.json")
        os.system(f"cp {meta_file} {export_data_dir}")
        meta_file = os.path.join(partnet_data_dir, data_id, "mobility_v2.json")
        os.system(f"cp {meta_file} {export_data_dir}")
        # [DEBUG]
        if debug:
            export_urdf_file = os.path.join(export_data_dir, urdf_name)
            urdf = URDF.load(export_urdf_file)
            urdf.show()

    # Rename after step 1.
    urdf_name = urdf_name.replace(".urdf", "_v1.urdf")

    ##################### Compute lgmcts templates ####################
    ids_compute = valid_ids
    for data_id in tqdm(ids_compute):
        partnet_dir = os.path.join(urdf_export_dir, data_id)
        create_yaml_from_partnet(partnet_dir, template_export_dir, data_id, urdf_file=urdf_name, verbose=False)
        template_file = os.path.join(template_export_dir, f"{data_id}.yaml")
        with open(template_file, "r") as f:
            template = yaml.load(f, Loader=yaml.FullLoader)
        # [DEBUG]
        if debug:
            art_obj = load_art_object_from_config(template)
            vis = art_obj.get_vis_o3d(show_joint=True)
            o3d.visualization.draw_geometries(vis)

    # ################### Post-modify the templates ####################
    # 1. For triggers that are missing handle. We need to add them manually.
    # Some objects are missing table top, we need to add them manually.
    print("Add handles to the URDF file.")
    for data_id in tqdm(ids_compute):
        urdf_export_data_dir = os.path.join(urdf_export_dir, data_id)
        os.makedirs(urdf_export_data_dir, exist_ok=True)
        template_file = os.path.join(template_export_dir, f"{data_id}.yaml")
        if not os.path.exists(template_file):
            continue
        with open(template_file, "r") as f:
            template = yaml.load(f, Loader=yaml.FullLoader)
        art_obj = load_art_object_from_config(template)
        art_obj.set_joint_values([0.0] * len(art_obj.active_joints))

        # [DEBUG]
        if debug:
            vis = art_obj.get_vis_o3d(show_joint=True)
            for space in art_obj.spaces.values():
                cur_vis = deepcopy(vis)
                vis += space.get_vis_o3d()
                o3d.visualization.draw_geometries(vis)
                vis = cur_vis

        link_names = []
        mesh_names = []
        texture_files = []
        mesh_files = []
        pose_xyzs = []
        pose_rpys = []
        for space in list(art_obj.spaces.values()):
            trigger_name = space.trigger[0][0]
            if trigger_name == "all":
                continue
            trigger_link = art_obj.links[trigger_name]
            trigger_raw_name = trigger_link.info["raw_name"]  # name in urdf
            trigger_joint = art_obj.joints[trigger_name]
            if len(trigger_link.handles) == 0:
                handle = create_handle(art_obj, trigger_link, trigger_joint)
                if handle is not None:
                    # Create bbox mesh: trimesh
                    texture_path = rng.choice(handle_texture)
                    print(texture_path)
                    ############################## Use BBox as handle ##############################
                    # handle_mesh = create_textured_cuboid(
                    #     sizes=handle.extent,
                    #     texture_path=texture_path,
                    # )
                    ############################## Load template handle ##############################
                    handle_mesh = resize_textured_obj(handle.extent, HANDLE_TEMPLATES[rng.randint(len(HANDLE_TEMPLATES))])
                    # # [DEBUG]
                    # scene = trimesh.scene.scene.Scene(geometry=[handle_mesh])
                    # scene.show()
                    # NOTICE: here the origin of the real link is at the joint
                    # This is the tf2parent w.r.t to the center of the link
                    # We need to convert it to w.r.t to the origin of the link
                    bbox_tf2parent = (
                        np.linalg.inv(trigger_link.get_pose(is_raw=True, return_mat=True))
                        @ trigger_link.get_pose(is_raw=False, return_mat=True)
                        @ handle.tf2parent
                    )
                    bbox_pose = bbox_tf2parent[:3, 3]
                    bbox_rpy = R.from_matrix(bbox_tf2parent[:3, :3]).as_euler("xyz")
                    # Save the bbox mesh to file
                    handle_mesh_export_dir = os.path.join(urdf_export_data_dir, "handle_meshes")
                    os.makedirs(handle_mesh_export_dir, exist_ok=True)
                    handle_mesh_file = os.path.join(handle_mesh_export_dir, f"{trigger_name}_{handle.name}.obj")
                    handle_mesh.export(handle_mesh_file)
                    bbox_mesh_rel_file = os.path.relpath(handle_mesh_file, urdf_export_data_dir)
                    link_names.append(trigger_raw_name)
                    mesh_names.append(handle.name)
                    texture_files.append(texture_path)
                    mesh_files.append(bbox_mesh_rel_file)
                    pose_xyzs.append(bbox_pose)
                    pose_rpys.append(bbox_rpy)

        # Create table top
        if data_id in storage_ids:
            # Locate the base link.
            base_link = art_obj.links["base"].children[0].child
            table_top = create_table_top(art_obj, base_link)
            if table_top is not None:
                # Create bbox mesh: trimesh
                texture_path = rng.choice(table_top_texture)
                print(texture_path)
                handle_mesh = create_textured_cuboid(
                    sizes=table_top.extent,
                    texture_path=texture_path,
                )
                bbox_tf2parent = (
                    np.linalg.inv(base_link.get_pose(is_raw=True, return_mat=True))
                    @ base_link.get_pose(is_raw=False, return_mat=True)
                    @ table_top.tf2parent
                )
                bbox_pose = bbox_tf2parent[:3, 3]
                bbox_rpy = R.from_matrix(bbox_tf2parent[:3, :3]).as_euler("xyz")
                # Save the bbox mesh to file
                table_top_export_dir = os.path.join(urdf_export_data_dir, "table_top_meshes")
                os.makedirs(table_top_export_dir, exist_ok=True)
                handle_mesh_file = os.path.join(table_top_export_dir, f"{base_link.info['raw_name']}_{table_top.name}.obj")
                handle_mesh.export(handle_mesh_file)
                box_mesh_rel_file = os.path.relpath(handle_mesh_file, urdf_export_data_dir)
                link_names.append(base_link.info["raw_name"])
                mesh_names.append(table_top.name)
                texture_files.append(texture_path)
                mesh_files.append(box_mesh_rel_file)
                pose_xyzs.append(bbox_pose)
                pose_rpys.append(bbox_rpy)
        # Add the meshes to the URDF file
        if len(link_names) > 0:
            urdf_file = os.path.join(urdf_export_dir, data_id, urdf_name)
            print(f"data_id: {data_id}, urdf_file: {urdf_file}")
            add_meshes_to_link(urdf_file, link_names, mesh_names, mesh_files, texture_files, pose_xyzs, pose_rpys)
            print(f"data_id: {data_id}, urdf_file: {urdf_file}")
            assert data_id in urdf_file, f"Data id {data_id} not in {urdf_file}."
            print(f"Added {len(link_names)} meshes to {urdf_file}'s {trigger_raw_name}, data_id: {data_id}.")
        else:
            # Copy the urdf file to urdf_modified_file
            urdf_file = os.path.join(urdf_export_dir, data_id, urdf_name)
            urdf_modified_file = urdf_file.replace("_v1.urdf", "_v2.urdf")
            os.system(f"cp {urdf_file} {urdf_modified_file}")

    # #################### Re-Compute lgmcts templates for modified ####################
    # left_models = os.listdir(urdf_export_dir)
    # left_models = [v for v in left_models if v.isdigit()]
    print("Re-Compute lgmcts templates for modified URDF.")
    for data_id in tqdm(ids_compute):
        partnet_dir = os.path.join(urdf_export_dir, data_id)
        create_yaml_from_partnet(
            partnet_dir,
            template_export_dir,
            data_id,
            urdf_file=urdf_name.replace("_v1.urdf", "_v2.urdf"),
            verbose=False,
        )
        # # Visualize raw urdf
        # urdf_modified_file = urdf_file.replace("_v1.urdf", "_v2.urdf")
        # urdf = URDF.load(urdf_modified_file)
        # urdf.show()
        if debug:
            # # [DEBUG]
            template_file = os.path.join(template_export_dir, f"{data_id}.yaml")
            print(template_file)
            with open(template_file, "r") as f:
                template = yaml.load(f, Loader=yaml.FullLoader)
            art_obj = load_art_object_from_config(template)
            vis = art_obj.get_vis_o3d(show_joint=True)
            for space in art_obj.spaces.values():
                vis += space.get_vis_o3d()
            o3d.visualization.draw_geometries(vis)
