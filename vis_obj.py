"""Visualize obj."""

import numpy as np
import open3d as o3d
import os
import yaml
from urchin import URDF
from lgmcts.sampler import Sampler, SceneState
from lgmcts.object_primitives import (
    ArtObject,
    Link,
    RigidObject,
    Space,
    load_art_object_from_config,
    load_link_from_config,
    load_space_from_config,
)


def visualize_all_arts(template_dir, urdf_dir, joint_values, is_robot=False, id=None):
    scale = 1.0
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    art_list = os.listdir(template_dir)
    for art_file in art_list:
        if id is not None:
            if art_file != f"{id}.yaml":
                continue
        print(f"Visualizing {art_file}")
        art_file = f"{template_dir}/{art_file}"
        with open(art_file, "r") as f:
            table_config = yaml.safe_load(f)
        art_obj = load_art_object_from_config(table_config, scale=scale, is_robot=is_robot)
        num_active_joints = len(art_obj.active_joints)
        print(art_obj.active_joints)
        if len(joint_values) == num_active_joints:
            art_obj.set_joint_values(joint_values, is_rel=True)
        elif len(joint_values) == 1:
            art_obj.set_joint_values([joint_values[0]] * num_active_joints, is_rel=True)
        else:
            _joint_values = [0.0] * num_active_joints
            _joint_values[: min(len(joint_values), num_active_joints)] = joint_values[: min(len(joint_values), num_active_joints)]
            art_obj.set_joint_values(_joint_values, is_rel=True)
        # art_obj.set_joint_values([0.5] * num_active_joints, is_rel=True)
        # art_obj.set_joint_values([0.0] * num_active_joints, is_rel=True)
        print(art_obj.get_joint_values(is_rel=True))
        vis = art_obj.get_vis_o3d(show_joint=True)
        for space in art_obj.spaces.values():
            vis += space.get_vis_o3d()
        o3d.visualization.draw_geometries(vis)

        robot = URDF.load(os.path.join(urdf_dir, art_obj.info["urdf_file"]))
        # # Set joint values
        # if is_robot:
        #     robot.set_joint_values(art_obj.get_joint_values(is_rel=True))
        #     robot.show()
        joint_cfg = art_obj.get_joint_values(is_rel=False, return_dict=True)
        joint_cfg = {art_obj.joints[j].info["raw_name"]: joint_cfg[j] for j in art_obj.active_joints}
        robot.show(cfg=joint_cfg, use_collision=False)


def visualie_rigid_obj(obj_file):
    rigid_obj = o3d.io.read_triangle_mesh(obj_file, True)
    rigid_obj.compute_vertex_normals()
    rigid_obj.paint_uniform_color([0.7, 0.7, 0.7])
    print(rigid_obj.has_textures())
    o3d.visualization.draw_geometries([rigid_obj])
    # # Use trimesh to load the obj file
    # import trimesh
    # obj_mesh = trimesh.load(obj_file)
    # obj_mesh.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_dir", type=str, default="./assets")
    parser.add_argument("--template_dir", type=str, default="art_templates")
    parser.add_argument("--urdf_dir", type=str, default="art_urdf")
    parser.add_argument("--is_robot", action="store_true")
    parser.add_argument("--joint_values", type=str, default="0.0")
    parser.add_argument("--id", type=int, default=12428)
    args = parser.parse_args()

    is_robot = args.is_robot
    assets_dir = args.assets_dir
    template_dir = os.path.join(assets_dir, args.template_dir)
    urdf_dir = os.path.join(assets_dir, args.urdf_dir)
    id = args.id
    joint_values = [float(x) for x in args.joint_values.split(",")]
    visualize_all_arts(template_dir, urdf_dir, joint_values, is_robot=is_robot, id=id)

    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # config_dir = os.path.join(cur_dir, "config")
    # task_file = os.path.join(config_dir, "tasks", "task_0.yaml")
    # task_root = os.path.join(cur_dir, "assets")
    # test_manip(task_file, task_root)
    # generate

    # rigid_asset_folder = os.path.join(cur_dir, "assets", "rigid_urdf")
    # obj_file = os.path.join(rigid_asset_folder, "test/test.obj")
    # visualie_rigid_obj(obj_file)
