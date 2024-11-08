"""Download and prepare objaverse data for usage."""

from tqdm import tqdm
import trimesh
import os
import json
import numpy as np
import random
import objaverse
import multiprocessing


def write_urdf_file(visual_file, collision_file, uid, export_dir, mass=0.1):
    """Write the URDF file."""
    urdf_file = os.path.join(export_dir, f"model.urdf")
    with open(urdf_file, "w") as f:
        f.write(f'<?xml version="1.0"?>\n')
        f.write(f'<robot name="{uid}">\n')
        f.write(f'  <link name="{uid}_link">\n')
        f.write(f"    <interial>\n")
        f.write(f'       <mass value="{mass}" />\n')
        f.write(f"    </interial>\n")
        f.write(f"    <visual>\n")
        f.write(f"      <geometry>\n")
        f.write(f'        <mesh filename="{visual_file}" />\n')
        f.write(f"      </geometry>\n")
        f.write(f"    </visual>\n")
        f.write(f"    <collision>\n")
        f.write(f"      <geometry>\n")
        f.write(f'        <mesh filename="{collision_file}" />\n')
        f.write(f"      </geometry>\n")
        f.write(f"    </collision>\n")
        f.write(f"  </link>\n")
        f.write(f"</robot>\n")
    return urdf_file


def prepare_lgmcts_template(mesh, data_name, mesh_file, template_dir, uid):
    bbox = mesh.bounding_box.bounds
    extents = bbox[1] - bbox[0]
    offset = (bbox[1] + bbox[0]) / 2
    rigid_template = {
        "name": data_name,
        "template_id": uid,
        "type": "RigidObject",
        "extent": [float(extents[0]), float(extents[1]), float(extents[2])],
        "offset": [float(offset[0]), float(offset[1]), float(offset[2])],
        "mass": 1.0,
        "info": {
            "mesh_file": mesh_file,
        },
    }
    with open(os.path.join(template_dir, f"{uid}.yaml"), "w") as f:
        json.dump(rigid_template, f)


def generate_rigid_urdf(
    objects,
    uid,
    uids_cat,
    export_dir,
    template_dir,
    max_size_edge=1.0,
    filter_thresh=0.2,
):
    """Generate rigid URDF files from objaverse data.
    The trick here is we will generate a block as its collision mesh.
    This is to make the objects easier to grasp.
    filter_thresh: float, the threshold to filter out the flat objects.
    """
    obj_file = objects[uid]
    export_uid_dir = os.path.join(export_dir, uid)
    export_uid_mesh_dir = os.path.join(export_uid_dir, "meshes")
    visual_file = os.path.join(export_uid_mesh_dir, f"{uid}_visual.obj")
    # Jump if already exists
    if os.path.exists(visual_file):
        return None
    mesh = trimesh.load(obj_file)
    if isinstance(mesh, trimesh.Scene):
        # Concatenate all meshes
        mesh = trimesh.util.concatenate(mesh.geometry.values())
    # Generate the collision mesh
    bbox = mesh.bounding_box
    bbox_extents = np.copy(bbox.extents)
    bbox_mesh = trimesh.creation.box(extents=bbox_extents)
    bbox_mesh.apply_transform(bbox.primitive.transform)
    # Compute the scale ratio
    scale = max_size_edge / (np.max(bbox_extents) + 1e-8)
    if np.min(bbox_extents) / (np.max(bbox_extents) + 1e-8) < filter_thresh:
        print(f"[Warning] Filtered out {uid} as it is too flat.")
        return None
    # Apply the scale
    mesh.apply_scale(scale)
    bbox_mesh.apply_scale(scale)
    # Move center to origin
    mesh.apply_translation(-bbox_mesh.centroid)
    bbox_mesh.apply_translation(-bbox_mesh.centroid)
    # Export the collision mesh
    if not os.path.exists(export_uid_mesh_dir):
        os.makedirs(export_uid_mesh_dir)
    collision_file = os.path.join(export_uid_mesh_dir, f"{uid}_collision.obj")
    bbox_mesh.export(collision_file)
    # visual_file = os.path.join(export_uid_mesh_dir, f"{uid}_visual.obj")
    mesh.export(visual_file)
    # # [DEBUG]
    # scene = trimesh.scene.scene.Scene(geometry=[mesh])
    # scene.show()
    # Generate the URDF file
    urdf_file = write_urdf_file(visual_file, collision_file, uid, export_uid_dir)
    # Generate the template for lgmcts
    prepare_lgmcts_template(mesh, uids_cat[uid], collision_file, template_dir, uid)
    return urdf_file


if __name__ == "__main__":
    category_list = [
        "apple",
        "ball",
        "pear",
        "peach",
        # "soap",
        "jam",
        "can",
        "teddy_bear",
        "potato",
        "milk",
        # "alcohol",
    ]  # Category we want to handle
    uids = objaverse.load_uids()
    lvis_annotations = objaverse.load_lvis_annotations()
    relv_uids = []  # Relevant uids
    uids_cat = {}
    cat2uids = {}
    max_per_cat = 2  # Maximum number of models per category
    for key in lvis_annotations.keys():
        if key in category_list:
            cat_uids = random.sample(lvis_annotations[key], min(max_per_cat, len(lvis_annotations[key])))
            relv_uids.extend(cat_uids)
            for uid in cat_uids:
                uids_cat[uid] = key
                if key not in cat2uids:
                    cat2uids[key] = []
                cat2uids[key].append(uid)

    print(f"Total number of models: {len(relv_uids)}")
    annotations = objaverse.load_annotations(relv_uids)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    export_dir = f"{cur_dir}/../assets/rigid_urdf/objaverse"
    template_dir = f"{cur_dir}/../assets/rigid_templates"
    asset_dir = os.path.join(cur_dir, "../assets")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(template_dir, exist_ok=True)
    processes = multiprocessing.cpu_count()
    objects = objaverse.load_objects(uids=relv_uids, download_processes=processes)

    started = False
    for i in tqdm(range(len(relv_uids))):
        uid = relv_uids[i]
        print(f"Processing {uid}")
        urdf_file = generate_rigid_urdf(objects, uid, uids_cat, export_dir, template_dir)

    # Save the uids_cat
    with open(f"{asset_dir}/objaverse_uids_cat.json", "w") as f:
        json.dump(uids_cat, f)
    # Save the cat2uids
    with open(f"{asset_dir}/objaverse_cat2uids.json", "w") as f:
        json.dump(cat2uids, f)
