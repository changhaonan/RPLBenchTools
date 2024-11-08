"""Isaaclab has a trouble in correctly loading the partnet urdf files.
1. In partnet-mobility, textures are sharing vertex. Isaaclab cannot correctly load the textures correctly.
"""

import trimesh
import os
import tqdm
import json
import numpy as np
from urchin import URDF
from PIL import Image


def parse_mtl_to_dict(mtl_content: str) -> dict:
    materials = {}
    current_material = None
    for line in mtl_content.splitlines():
        line = line.strip()
        if line.startswith("newmtl"):
            current_material = line.split()[1]
            materials[current_material] = ""
        elif line.startswith("map_Kd") and current_material:
            materials[current_material] = line.split()[1]
    return materials


def remove_material_from_obj(obj_content: str, materials_to_remove: list) -> str:
    result = []
    skip_material = False

    for line in obj_content.splitlines():
        line = line.strip()
        if line.startswith("usemtl"):
            current_material = line.split()[1]
            if current_material in materials_to_remove:
                skip_material = True
            else:
                skip_material = False
        if not skip_material:
            result.append(line)

    return "\n".join(result)


def deep_fix_urdf_v2(data_id, urdf_name, urdf_root, fixed_urdf_root):
    urdf_dir = os.path.join(urdf_root, data_id)
    fixed_urdf_dir = os.path.join(fixed_urdf_root, data_id)
    os.makedirs(fixed_urdf_dir, exist_ok=True)
    os.makedirs(os.path.join(fixed_urdf_dir, "textured_objs"), exist_ok=True)
    # search all textured objects
    textured_objs = os.listdir(os.path.join(urdf_dir, "textured_objs"))
    textured_objs = [x for x in textured_objs if x.endswith(".obj")]

    for textured_obj in textured_objs:
        obj_name = textured_obj.split(".")[0]
        mtl_file = os.path.join(urdf_dir, "textured_objs", f"{obj_name}.mtl")
        obj_file = os.path.join(urdf_dir, "textured_objs", textured_obj)
        export_obj_file = os.path.join(fixed_urdf_dir, "textured_objs", textured_obj)
        # parse mtl file
        with open(mtl_file, "r") as f:
            mtl_content = f.read()
        materials = parse_mtl_to_dict(mtl_content)
        # remove materials when there are more than one texture, and some textures have image but some not
        materials_to_remove = []
        if any([texture != "" for texture in materials.values()]):
            for material, texture in materials.items():
                if texture == "":
                    materials_to_remove.append(material)
        # remove materials from obj file
        with open(obj_file, "r") as f:
            obj_content = f.read()
        obj_content = remove_material_from_obj(obj_content, materials_to_remove)
        # export obj file
        with open(export_obj_file, "w") as f:
            f.write(obj_content)
        # copy mtl file
        os.system(f"cp {mtl_file} {os.path.join(fixed_urdf_dir, 'textured_objs')}")
    # Copy images folder
    os.system(f"cp -r {os.path.join(urdf_dir, 'images')} {fixed_urdf_dir}")
    # Copy all .txt, .json, .txt file
    for file in os.listdir(urdf_dir):
        if file.endswith(".txt") or file.endswith(".json") or file.endswith(".txt") or file.endswith(".urdf"):
            os.system(f"cp {os.path.join(urdf_dir, file)} {os.path.join(fixed_urdf_dir, file)}")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default="./raw_data/dataset")
    argparser.add_argument("--data_name", type=str, default="")
    argparser.add_argument("--task_type", type=str, default="all", help="task_20, all")
    args = argparser.parse_args()

    data_dir = args.data_dir
    fixed_urdf_root = data_dir.replace("dataset", "dataset_fixed")
    task_type = args.task_type
    # Tasks
    if task_type == "task_20":
        task_ids = [
            "45961",
            "19179",
            "44853",
            "35059",
            "45746",
            "45132",
            "45159",
            "7310",
            "45176",
            "45244",
            "45249",
            "45323",
            "45305",
            "12480",
            "45427",
            "7290",
            "45523",
            "45676",
            "45623",
            "45645",
        ]
    else:
        task_ids = os.listdir(data_dir)
    # filter non-numeric folders
    task_ids = [x for x in task_ids if x.isdigit()]
    for data_id in tqdm.tqdm(task_ids):
        deep_fix_urdf_v2(data_id, "mobility.urdf", data_dir, fixed_urdf_root)
