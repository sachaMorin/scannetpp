import open3d as o3d
from pathlib import Path

data_path_v1_base = Path("/data/concept-graphs/scannetpp/data")
data_path_v2_base = Path("/data/concept-graphs/scannetpp_v2/data")


def load_mesh(mesh_file, color=None):
    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        mesh.compute_vertex_normals()
        if color:
            mesh.paint_uniform_color(color)  # Set a uniform color for distinction
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {mesh_file}, {e}")
        return None


def visualize_meshes(scene_id):
    data_path_v1 = data_path_v1_base / scene_id / "scans"
    data_path_v2 = data_path_v2_base / scene_id / "scans"

    mesh_file_v1 = data_path_v1 / "mesh_aligned_0.05.ply"
    mesh_file_v2 = data_path_v2 / "mesh_aligned_0.05.ply"

    mesh_v1 = load_mesh(mesh_file_v1, color=[1, 0, 0])  # Red for version 1
    mesh_v2 = load_mesh(mesh_file_v2, color=[0, 1, 0])  # Green for version 2

    if not mesh_v1 or not mesh_v2:
        print(f"Failed to load one or both meshes for scene_id: {scene_id}")
        return

    o3d.visualization.draw_geometries(
        [mesh_v1, mesh_v2],
        window_name=f"Meshes for scene_id: {scene_id}",
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )


scene_id = "410c470782"
visualize_meshes(scene_id)
