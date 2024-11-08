"""Functions related to tasks"""

import yaml
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from lgmcts.object_primitives import (
    ArtObject,
    Link,
    Robot,
    RigidObject,
    Space,
    load_art_object_from_config,
    load_link_from_config,
    load_space_from_config,
    check_collision_SAT,
)
from lgmcts.utils import DRAWER_NAMES, DOOR_NAMES, BUTTON_NAMES, HANG_CATS


#################### Task loading method ##################
class Task:
    def __init__(self, task_des, robot, art_list, rigid_list, external_spaces, info):
        self.task_des: dict = task_des
        self.robot: Robot = robot
        self.art_list: list[ArtObject] = art_list
        self.rigid_list: list[RigidObject] = rigid_list
        self.external_spaces: list[Space] = external_spaces
        self.info: dict = info

    @property
    def cam_dict(self):
        _cam_dict = {}
        for key in self.info["camera"].keys():
            _cam_dict[key] = self.info["camera"][key]
            # tolist
            for k in _cam_dict[key].keys():
                _cam_dict[key][k] = _cam_dict[key][k].tolist()
        return _cam_dict


def load_task_from_config_file(asset_dir, config_file):
    with open(config_file, "r") as f:
        task_config = yaml.safe_load(f)
    task = task_config.get("task", {})
    robot_type = task_config["robot_type"]
    robot_scale = task_config.get("robot_scale", 1.0)
    # Init robot
    robot_file = os.path.join(asset_dir, "robot_templates", f"{robot_type}.yaml")
    with open(robot_file, "r") as f:
        robot_config = yaml.safe_load(f)
    robot = load_art_object_from_config(robot_config, robot_scale, is_robot=True)

    # Init scene
    art_list = []
    rigid_list = []
    external_spaces = {}
    for config in task_config["objects"]:
        if config["type"] == "ArtObject":
            if "load_from_template" in config:
                template_file = os.path.join(asset_dir, "art_templates", f"{config['load_from_template']}.yaml")
                if not os.path.exists(template_file):
                    raise ValueError(f"Template file {template_file} does not exist.")
                else:
                    with open(template_file, "r") as f:
                        _obj_config = yaml.safe_load(f)
                    art_obj = load_art_object_from_config(_obj_config, config.get("scale", 1.0))
                    art_obj.template_id = config["load_from_template"]
            else:
                art_obj = load_art_object_from_config(config, config.get("scale", 1.0))
            # Offset to table-plane
            # TODO: for now, preserve it, but it is not necessary.
            raise_to_table = config.get("raise_to_table", True)
            if raise_to_table:
                bbox_min, bbox_max = art_obj.get_whole_bbox()
                offset_tf = np.zeros(6)
                offset_tf[2] = -bbox_min[2]
                art_obj.update_pose(offset_tf)
            # Update tf
            if "tf" in config:
                art_obj.update_pose(np.array(config["tf"]), move_center=False)
            # Update joint_values
            if "joint_values" in config:
                art_obj.set_joint_values(np.array(config["joint_values"]))
            art_list.append(art_obj)
        elif config["type"] == "RigidObject":
            if "load_from_template" in config:
                template_file = os.path.join(
                    asset_dir,
                    "rigid_templates",
                    f"{config['load_from_template']}.yaml",
                )
                if not os.path.exists(template_file):
                    raise ValueError(f"Template file {template_file} does not exist.")
                else:
                    with open(template_file, "r") as f:
                        _obj_config = yaml.safe_load(f)
                    rigid_obj = RigidObject(**load_link_from_config(_obj_config, config.get("scale", 1.0)))
                    rigid_obj.template_id = config["load_from_template"]
            else:
                rigid_obj = RigidObject(**load_link_from_config(config, config.get("scale", 1.0)))
            # Update tf
            if "tf" in config:
                rigid_obj.update_pose(np.array(config["tf"]))
            rigid_list.append(rigid_obj)
        elif config["type"] == "Space":
            space = Space(**load_space_from_config(config, config.get("scale", 1.0)))
            external_spaces[config["name"]] = space
    # Load camera setup
    info = {"camera": {}}
    if "camera" in task_config:
        camera_config = task_config["camera"]
        for key in camera_config.keys():
            camera_pose = np.array(camera_config[key]["pose"])
            camera_target = np.array(camera_config[key]["target"])
            info["camera"][key] = {"pose": camera_pose, "target": camera_target}
    return Task(task, robot, art_list, rigid_list, external_spaces, info)


#################### Task generation method ##############
def generate_pose_setup(art_list, rigid_list, cam_dict, task_level, rng, update_art=True, update_rigid=True, update_camera=True, **kwargs):
    # Parameters
    max_size_edge = kwargs.get("max_size_edge", 0.5)
    object_init_margin = kwargs.get("object_init_margin", 0.05)
    rigid_obj_scale = kwargs.get("rigid_obj_scale", 0.3)
    filter_out_small_space = kwargs.get("filter_out_small_space", True)
    ahead_offset = kwargs.get("ahead_offset", 0.6)
    lift_height_common = kwargs.get("lift_height_common", 0.2)
    lift_height_hang = kwargs.get("lift_height_hang", 0.5)
    max_sample_time = kwargs.get("max_sample_time", 5)
    size_threshold = object_init_margin + rigid_obj_scale

    # Generate poses
    pose_back = np.eye(4)
    pose_back[0, 3] = -max_size_edge / 2.0 - ahead_offset
    pose_back[1, 3] = 0
    pose_back[2, 3] = max_size_edge
    pose_back[:3, :3] = R.from_euler("z", 180, degrees=True).as_matrix()
    space_back = Space(
        name="buffer_back",
        tf=pose_back,
        extent=[max_size_edge, max_size_edge, 2.0 * max_size_edge],
    )

    # Sample pose for each art_obj
    if update_art:
        art_obj_space = [space_back]
        assert len(art_list) <= len(art_obj_space), f"More art_objs than spaces for task."
        for idx in range(len(art_list)):
            art_obj = art_list[idx]
            art_bbox_min, art_bbox_max = art_obj.get_whole_bbox()
            art_bbox_extent = art_bbox_max - art_bbox_min
            # Sample poses
            tf_in_spaces = art_obj_space[idx].sample_poses_w_rot(
                art_bbox_extent,
                n_samples=1,
                rng=rng,
                ignore_size=False,
                place_bottom=True,
            )
            # lift based on object type
            if any([cat_name in art_obj.obj_name for cat_name in HANG_CATS]):
                # Hang objects up
                tf_in_spaces[0][2, 3] += lift_height_hang
            else:
                tf_in_spaces[0][2, 3] += lift_height_common
            if len(tf_in_spaces) == 0:
                print("[Warning] No valid poses sampled.")
                return None
            art_obj.update_pose(tf_in_spaces[0])

    # Sample pose for rigid_obj
    if task_level != "easy" and task_level != "basic":
        # Level that are harder than easy can have objects started from inside.
        internal_spaces = []
        for art_obj in art_list:
            internal_spaces += list(art_obj.get_spaces(size_threshold=size_threshold).values())
        # Balancing the number of inside and outside
        if len(internal_spaces) > len(external_spaces):
            duplicate_ratio = int(len(internal_spaces) / len(external_spaces))
        else:
            duplicate_ratio = 1
        spaces_rigid_init = external_spaces * duplicate_ratio + internal_spaces
    else:
        internal_spaces = []
        for art_obj in art_list:
            internal_spaces += list(art_obj.get_spaces(size_threshold=size_threshold).values())
        spaces_table_top = [space for space in internal_spaces if "table-top" in space.name]
        # Init at table top for simple tasks
        spaces_rigid_init = spaces_table_top

    if update_rigid:
        spaces = [space_back]
        selected_spaces = rng.choice(spaces, len(art_list), replace=False)
        external_spaces = [space for space in spaces if space not in selected_spaces]
        # assert len(spaces_outside) != 0, "No spaces left!"
        if filter_out_small_space:
            size_threshold = object_init_margin + rigid_obj_scale  # object size + margin
        else:
            size_threshold = 0.0
        # [DEBUG]
        moved_rigid_list = []
        existing_bboxes = []
        for idx in range(len(rigid_list)):
            rigid_obj = rigid_list[idx]
            rigid_extent = rigid_obj.extent + object_init_margin
            tf_in_space = []  # Empty
            for _idx in range(max_sample_time):
                # Sample poses
                space_init = rng.choice(spaces_rigid_init, 1)[0]
                tf_in_space = space_init.sample_poses_w_rot(
                    rigid_extent,
                    n_samples=1,
                    rng=rng,
                    ignore_size=False,
                    place_bottom=True,
                )
                if len(tf_in_space) > 0:
                    rigid_obj.update_pose(tf_in_space[0])
                    # Check if in collision with other objects
                    cur_bbox = rigid_obj.get_bbox()
                    cur_bbox[2] += object_init_margin  # Pad Margin
                    if len(existing_bboxes) != 0:
                        collisions = check_collision_SAT(cur_bbox, existing_bboxes)
                        if any(collisions):
                            continue
                    existing_bboxes.append(cur_bbox)
                    moved_rigid_list.append(rigid_obj)  # Add obj to new
                    break
            # TODO: Test collisions with existing ones
    else:
        moved_rigid_list = rigid_list
        external_spaces = [space_back]

    if update_camera:
        camera_keys = ["front"]
        # poses for front
        cam_dist = 2.2727272727272725
        cam_y_shifts = np.random.rand(1) * 2.0 - 1.0  # (-1.0, 1.0)
        cam_z_shifts = np.random.rand(1) * 1.0  # (0.0, 1.0)
        cam_xs = np.sqrt((cam_dist) ** 2 - cam_y_shifts**2 - cam_z_shifts**2)
        cam_xs = np.clip(cam_xs, a_min=1.5, a_max=None)  # Clamping to ensure minimum value of 1.5
        cam_pos = np.stack((cam_xs, cam_y_shifts, 2.0 + cam_z_shifts), axis=1).tolist()
        cam_target = art_list[0].get_pose(is_raw=False, return_mat=False)[:3].tolist()
        cam_dict = {"front": {"pose": cam_pos, "target": cam_target}}
    else:
        pass
    return art_list, moved_rigid_list, internal_spaces, cam_dict


def generate_meta_task(
    template_dir,
    export_dir,
    robot_type,
    art_template_list,
    rigid_template_list,
    cat2uids,
    task_level,
    max_size_edge=0.5,
    min_size_edge=0.4,
    rigid_obj_scale=0.3,
    robot_scale=1.0,
    seed=0,
    chat_api=None,
):
    """
    We use a hierachical generation process. First we generate the major setup. After the
    Args:
        seed: random seed for setup.
        max_size_edge: maximum size of the edge of the object (horizontal).
    """
    rng = np.random.RandomState(seed=seed)
    # Select meta config based on task level
    if task_level == "easy" or task_level == "basic":
        num_rigid_obj = 2
        num_art_obj = 1
    elif task_level == "medium":
        num_rigid_obj = 2
        num_art_obj = 1
    elif task_level == "hard":
        num_rigid_obj = rng.randint(3, 6)  # [3, 5]
        num_art_obj = 2
    else:
        raise ValueError(f"Uknown level: {task_level}")

    # Select art_objs and rigid_objs
    max_try = 10
    for _ in range(max_try):
        # art_obj_templates = rng.choice(art_template_list, num_art_obj, replace=False)
        art_obj_templates = [art_template_list[seed % len(art_template_list)]]  # deterministic
        rigid_obj_cats = rng.choice(list(cat2uids.keys()), num_rigid_obj, replace=False)
        rigid_obj_templates = []
        for cat in rigid_obj_cats:
            rigid_obj_templates.append(rng.choice(cat2uids[cat]))
        # Load objs
        art_objs = []
        rigid_objs = []
        art_obj_load_status = True
        for template in art_obj_templates:
            with open(os.path.join(template_dir, "art_templates", template), "r") as f:
                cfg = yaml.safe_load(f)
            if not cfg["info"].get("load_status", True):
                art_obj_load_status = False
            art_objs.append(load_art_object_from_config(cfg))
        if art_obj_load_status:
            # If all art_objs are loaded successfully, break
            break
    if not art_obj_load_status:
        raise ValueError("Articulated objects are not loaded correctly.")
    for template in rigid_obj_templates:
        with open(os.path.join(template_dir, "rigid_templates", f"{template}.yaml"), "r") as f:
            cfg = yaml.safe_load(f)
        rigid_objs.append(RigidObject(**load_link_from_config(cfg)))

    # Compute scale bounds
    art_scale_up_bounds = []
    art_scale_down_bounds = []
    for art_obj in art_objs:
        bbox_min, bbox_max = art_obj.get_whole_bbox()
        scale = max_size_edge / np.max(bbox_max[:2] - bbox_min[:2])
        art_scale_up_bounds.append(scale)
        scale = min_size_edge / np.max(bbox_max[:2] - bbox_min[:2])
        art_scale_down_bounds.append(scale)

    # Scales
    rigid_scales = [rigid_obj_scale] * len(rigid_objs)
    art_scales = np.ones(len(art_objs)) * 0.8  # HACK: for now, we do not scale the art_objs
    # Export results
    output_dict = {}
    output_dict["level"] = task_level
    output_dict["seed"] = seed
    output_dict["robot_type"] = robot_type
    output_dict["robot_scale"] = robot_scale
    output_dict["objects"] = []
    # Add art objs
    for idx, art_obj in enumerate(art_objs):
        output_dict["objects"].append(
            {
                "name": str(art_obj.obj_name),
                "type": "ArtObject",
                "load_from_template": str(art_obj_templates[idx].split(".")[0]),
                "scale": float(art_scales[idx]),
            }
        )
    # Add rigid objs
    for idx, rigid_obj in enumerate(rigid_objs):
        output_dict["objects"].append(
            {
                "name": str(rigid_obj.name),
                "type": "RigidObject",
                "load_from_template": str(rigid_obj_templates[idx].split(".")[0]),
                "scale": float(rigid_scales[idx]),
            }
        )

    # Export meta task
    os.makedirs(os.path.join(export_dir, "meta"), exist_ok=True)
    with open(os.path.join(export_dir, "meta", f"meta_{task_level}_{seed:04}.yaml"), "w") as f:
        yaml.dump(output_dict, f)


def generate_full_task(
    export_dir,
    task_level,
    meta_task_id,
    max_size_edge,
    object_init_margin,
    rigid_obj_scale,
    seed,
    seed_offset=0,
    randomize_art=False,
    randomize_rigid=False,
    randomize_camera=False,
    is_ref=False,
    filter_out_small_space=True,
    debug=True,
    chat_api=None,
):
    # Generate detailed task with poses
    # Currently, we are using a very structured way to generate the set-up.
    # We devide the space into three spaces: 1. Left 2.Front 3.Right.
    # We put one or two articulated objects within two spaces. And use the rest of buffer space.
    # The whole region is a long rectangle of 1x3. Each space is of size 1x1.
    # There can be more subtle ways to do so.
    max_sample_time = 5
    rng = np.random.RandomState(seed=seed)
    # Load meta task first
    ref_config_file = os.path.join(export_dir, "ref", f"full_{task_level}_{meta_task_id:04}_ref.yaml")
    meta_config_file = os.path.join(export_dir, "meta", f"meta_{task_level}_{meta_task_id:04}.yaml")
    asset_dir = os.path.dirname(export_dir)
    if os.path.exists(os.path.join(export_dir, ref_config_file)) and not is_ref:
        # Load the ref config file
        task = load_task_from_config_file(asset_dir, ref_config_file)
    else:
        # Load meta task if not exist
        task = load_task_from_config_file(asset_dir, meta_config_file)
        randomize_art = True
        randomize_rigid = True
        randomize_camera = True
    robot, art_list, rigid_list, external_spaces, cam_dict = task.robot, task.art_list, task.rigid_list, task.external_spaces, task.cam_dict
    # Generate poses.
    art_list, rigid_list, internal_spaces, cam_dict = generate_pose_setup(
        art_list=art_list,
        rigid_list=rigid_list,
        task_level=task_level,
        cam_dict=cam_dict,
        rng=rng,
        update_art=randomize_art,
        update_rigid=randomize_rigid,
        update_camera=randomize_camera,
        max_size_edge=max_size_edge,
        object_init_margin=object_init_margin,
        rigid_obj_scale=rigid_obj_scale,
        filter_out_small_space=filter_out_small_space,
        max_sample_time=max_sample_time,
    )
    if debug:
        print("Articulated objects:")
        for art_obj in art_list:
            print(art_obj.obj_name)
        # [DEBUG]
        import open3d as o3d

        vis = []
        for art_obj in art_list:
            vis += art_obj.get_vis_o3d()
        for rigid_obj in rigid_list:
            vis += rigid_obj.get_vis_o3d()
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis += [origin]
        # # Add spaces
        # for space in spaces_rigid_init:
        #     vis += space.get_vis_o3d()
        # o3d.visualization.draw_geometries(vis)

        # Add z=0 surface
        plane = o3d.geometry.TriangleMesh.create_box(1, 1, 0.005)
        plane.compute_vertex_normals()
        plane.paint_uniform_color([0.7, 0.7, 0.7])
        plane.translate([0, 0, -0.005])
        vis += [plane]
        o3d.visualization.draw_geometries(vis)

    ####################################### Generate task desc ########################################
    if chat_api is not None:
        # Use api to generate the description.
        # Prompt engineering.
        pass
    else:
        # Use heuristic to generate.
        art_cats = [art_obj.obj_name.split("_")[0] for art_obj in art_list]
        goal_spaces = [art_obj.get_spaces(exclude=["table-top"], size_threshold=(object_init_margin + rigid_obj_scale)) for art_obj in art_list]

        rigid_cats = [rigid_obj.name for rigid_obj in rigid_list]  # Rigid name is type
        if task_level == "basic":
            # Basic skills
            task_description = f"Open the {art_cats[0]}."
            valid_joints = []
            for j in art_list[0].active_joints:
                if any([drawer_name in j for drawer_name in DRAWER_NAMES]) or any([door_name in j for door_name in DOOR_NAMES]):
                    if any([button_name in j for button_name in BUTTON_NAMES]):
                        continue
                    valid_joints.append(j)
            goals_config = [
                {
                    "manip_obj": art_cats[0],
                    "manip_joint": str(rng.choice(valid_joints, 1)[0]),
                    "relation": "open",
                }
            ]

        elif task_level == "easy" or task_level == "medium":
            task_description = f"Put the {rigid_cats[0]} into the {art_cats[0]}."
            goals_config = [
                {
                    "manip_obj": rigid_cats[0],
                    "goal_obj": art_cats[0],
                    "goal_space": str(rng.choice(list(goal_spaces[0].keys()), 1)[0]),
                    "relation": "in",
                }
            ]
        elif task_level == "hard":
            # For hard task there is two goals
            task_description = ""
            goals_config = []
            for _idx in range(len(art_cats)):
                task_description += f"Put the {rigid_cats[_idx]} into the {art_cats[_idx]}. "
                goal_config = {
                    "manip_obj": rigid_cats[_idx],
                    "goal_obj": art_cats[_idx],
                    "goal_space": str(rng.choice(list(goal_spaces[_idx].keys()), 1)[0]),
                    "relation": "in",
                }
                goals_config.append(goal_config)
        else:
            raise NotImplementedError(f"{task_level} not implemented.")

    # Export the everything
    export_dict = {}
    export_dict["task"] = {"description": task_description, "goals": goals_config}
    export_dict["level"] = task_level
    export_dict["seed"] = seed
    export_dict["robot_type"] = robot.obj_name
    export_dict["robot_scale"] = robot.scale
    export_dict["objects"] = []
    # Add articulated
    for art_obj in art_list:
        export_dict["objects"].append(art_obj.get_cfg())
    # Add rigid
    for rigid_obj in rigid_list:
        export_dict["objects"].append(rigid_obj.get_cfg())
    # Add outside space
    for idx, space in enumerate(external_spaces):
        export_dict["objects"].append(space.get_cfg(name=f"buffer_{idx}" if not space.name else space.name))
    export_dict["camera"] = cam_dict
    # Export
    if not is_ref:
        task_file = os.path.join(export_dir, f"full_{task_level}_{meta_task_id:04}_{seed:06}.yaml")
    else:
        task_file = os.path.join(export_dir, ref_config_file)
    with open(task_file, "w") as f:
        yaml.dump(export_dict, f)
    return task_file
