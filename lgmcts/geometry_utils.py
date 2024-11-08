import trimesh
import numpy as np
from PIL import Image


def create_textured_cuboid(sizes, texture_path=None):
    size_x, size_y, size_z = sizes
    # Define uv coordinates.
    dx, dy = 0.5 / 2, 0.66 / 2
    uvs = np.array(
        [
            [dx, 0],
            [2 * dx, 0],
            [0, dy],
            [dx, dy],
            [2 * dx, dy],
            [3 * dx, dy],
            [1, dy],
            [0, 2 * dy],
            [dx, 2 * dy],
            [2 * dx, 2 * dy],
            [3 * dx, 2 * dy],
            [1, 2 * dy],
            [dx, 1],
            [2 * dx, 1],
        ]
    )  # (14, 2)

    # Define the vertices and triangles of the cube
    vertices = np.array(
        [
            [0, 0, 0],
            [0, size_y, 0],
            [size_x, size_y, 0],
            [size_x, 0, 0],
            [0, 0, size_z],
            [0, size_y, size_z],
            [size_x, size_y, size_z],
            [size_x, 0, size_z],
        ]
    )  # (8, 3)
    # shift the vertices to the center
    vertices -= np.array([size_x / 2, size_y / 2, size_z / 2])

    triangles = np.array(
        [[0, 1, 2], [0, 2, 3], [5, 6, 1], [1, 6, 2], [3, 2, 6], [3, 6, 7], [3, 4, 0], [7, 4, 3], [4, 5, 1], [0, 4, 1], [6, 5, 7], [7, 5, 4]]
    )  # 12 triangles. (12, 3)

    # The order must follow the order of triangles.
    triangles_uvs = np.array(
        [
            uvs[8],
            uvs[9],
            uvs[4],
            uvs[8],
            uvs[4],
            uvs[3],
            uvs[10],
            uvs[5],
            uvs[9],
            uvs[9],
            uvs[5],
            uvs[4],
            uvs[3],
            uvs[4],
            uvs[1],
            uvs[3],
            uvs[1],
            uvs[0],
            uvs[3],
            uvs[7],
            uvs[8],
            uvs[2],
            uvs[7],
            uvs[3],
            uvs[12],
            uvs[13],
            uvs[9],
            uvs[8],
            uvs[12],
            uvs[9],
            uvs[5],
            uvs[10],
            uvs[6],
            uvs[6],
            uvs[10],
            uvs[11],
        ]
    )  # (3 * num_triangles, 2)
    assert triangles_uvs.shape == (36, 2)

    # create trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # mesh.visual.uv = uvs
    # mesh.visual.face_uvs = triangles_uvs

    texture_image = Image.open(texture_path)
    texture = trimesh.visual.TextureVisuals(uv=triangles_uvs, image=texture_image)
    mesh.visual = texture
    # mesh.show()
    return mesh


def resize_textured_obj(sizes, obj_path):
    mesh = trimesh.load_mesh(obj_path)
    extent = mesh.extents
    long_axis = np.argmax(extent[:2])
    if sizes[long_axis] < sizes[1 - long_axis]:
        # Meaning we need to rotate the mesh by 90 degrees
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
        extent = [extent[1], extent[0], extent[2]]
    # bounds = mesh.bounds
    # min_z = bounds[0, 2]
    scales = sizes / extent
    mesh.apply_scale(scales)
    center = mesh.centroid
    bounds = mesh.bounds
    mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0, 2]])
    return mesh


# Load the texture image
#
# mesh.textures = [o3d.geometry.Image(texture_image)]

# # Compute normals for better visualization
# mesh.compute_vertex_normals()

# # Visualize the mesh
# o3d.visualization.draw_geometries([mesh])
if __name__ == "__main__":
    create_textured_cube(sizes=[1.0, 0.1, 1.0])
