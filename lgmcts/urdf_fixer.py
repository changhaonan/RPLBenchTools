import re
import os
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET


def fix_urdf_field(file_path, version_id=0):
    try:
        with open(file_path, "r") as file:
            modified_lines = []
            for line in file:
                if line.strip().startswith("<limit"):
                    # Add 'effort' and 'velocity' attribute if not present
                    if ("effort=" not in line) or ("velocity=" not in line):
                        line = re.sub(r"(<limit)(.*?>)", r'\1 effort="30000" velocity="1000.0"\2', line)
                    modified_lines.append(line)
                else:
                    modified_lines.append(line)
            # Replace the None with 0
            modified_lines = [re.sub(r"None", r"0", x) for x in modified_lines]
        # Write the modified lines to a new file
        with open(file_path, "w") as file:
            file.writelines(modified_lines)
        return True

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def remove_nonexistent_files_from_urdf(urdf_file, urdf_folder):
    # Parse the URDF file
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Loop through all 'visual' tags and check if the mesh file exists
    for link in root.findall(".//link"):
        for visual in link.findall(".//visual"):
            mesh = visual.find(".//mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                if filename and not os.path.exists(os.path.join(urdf_folder, filename)):  # Check if the file doesn't exist
                    link.remove(visual)  # Remove visual element from its parent link

    # Loop through all 'collision' tags and check if the mesh file exists
    for link in root.findall(".//link"):
        for collision in link.findall(".//collision"):
            mesh = collision.find(".//mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                if filename and not os.path.exists(os.path.join(urdf_folder, filename)):  # Check if the file doesn't exist
                    link.remove(collision)  # Remove collision element from its parent link

    # Write the modified URDF back to the file
    tree.write(urdf_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default="./raw_data/dataset")
    argparser.add_argument("--data_name", type=str, default="")
    argparser.add_argument("--urdf_name", type=str, default="mobility.urdf")
    argparser.add_argument("--version", type=int, default=0, help="Version of the dataset")
    args = argparser.parse_args()
    folder_path = args.data_dir
    version_id = int(args.version)
    urdf_name = args.urdf_name
    if args.data_name:
        fix_urdf_field(f"{folder_path}/{args.data_name}/{urdf_name}", version_id=version_id)
    else:
        valid_dataset_idx_file = f"valid_dataset_idx_v{version_id}.txt"
        valid_dataset_idx_file_full_path = os.path.join(folder_path, valid_dataset_idx_file)
        if os.path.isfile(valid_dataset_idx_file_full_path):
            with open(valid_dataset_idx_file_full_path, "r") as file:
                valid_dataset_idx = file.read().splitlines()

            valid_dataset_idx = [int(x) for x in valid_dataset_idx if x]
        else:
            valid_dataset_idx = []

        dataset_idxs = [int(x) for x in os.listdir(folder_path) if x.isdigit()]

        # avoid duplicates
        dataset_idxs = [str(x) for x in dataset_idxs if x not in valid_dataset_idx]
        # dataset_idxs = ["7130"]
        for data_name in tqdm(dataset_idxs):
            if os.path.isdir(os.path.join(folder_path, data_name)):
                status = fix_urdf_field(f"{folder_path}/{data_name}/{urdf_name}", version_id=version_id)
                remove_nonexistent_files_from_urdf(f"{folder_path}/{data_name}/{urdf_name}", f"{folder_path}/{data_name}")
            else:
                continue
            if status:
                valid_dataset_idx.append(data_name)

        print(f"Valid dataset size: {len(valid_dataset_idx)}")
        with open(valid_dataset_idx_file_full_path, "w") as file:
            file.write("\n".join([str(x) for x in valid_dataset_idx]))
