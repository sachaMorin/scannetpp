from pathlib import Path
import numpy as np
import json
import open3d as o3d

# Define the paths and scene IDs
scene_ids = [
    # "0a76e06478",
    # "1f7cbbdde1",
    # "49a82360aa",
    # "8a35ef3cfe",
    "d918af9c5f",
    # "0a7cc12c0e",
    "410c470782",
    # "4c5c60fa76",
    # "c0f5742640",
    # "fd361ab85f",
]

data_path_v1_base = Path("/data/concept-graphs/scannetpp/data")
data_path_v2_base = Path("/data/concept-graphs/scannetpp_v2/data")


# Function to load mesh vertices
def load_mesh_vertices(mesh_file):
    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        return np.asarray(mesh.vertices)
    except FileNotFoundError:
        return None


# Function to load segment indices
def load_seg_indices(seg_indices_file):
    try:
        with open(seg_indices_file, "r") as f:
            return np.array(json.load(f)["segIndices"])
    except FileNotFoundError:
        return None


# Function to load segment annotations (labels and segments)
def load_seg_annotations(seg_anno_file):
    try:
        with open(seg_anno_file, "r") as f:
            seg_anno = json.load(f)["segGroups"]
        labels = [item["label"] for item in seg_anno]
        segments = [item["segments"] for item in seg_anno]
        return labels, segments
    except FileNotFoundError:
        return None, None


# Compare data for each scene ID
for scene_id in scene_ids:
    print(f"\nChecking scene_id: {scene_id}")
    data_path_v1 = data_path_v1_base / scene_id / "scans"
    data_path_v2 = data_path_v2_base / scene_id / "scans"

    # Compare `mesh_aligned_0.05.ply` (vertices)
    mesh_file_v1 = data_path_v1 / "mesh_aligned_0.05.ply"
    mesh_file_v2 = data_path_v2 / "mesh_aligned_0.05.ply"
    vertices_v1 = load_mesh_vertices(mesh_file_v1)
    vertices_v2 = load_mesh_vertices(mesh_file_v2)

    if vertices_v1 is None or vertices_v2 is None:
        print("  mesh_aligned_0.05.ply: File not found")
    elif np.array_equal(vertices_v1, vertices_v2):
        print("  mesh_aligned_0.05.ply: Vertices match")
    else:
        print("  mesh_aligned_0.05.ply: Vertices do not match")
        print("  shape v1: ", vertices_v1.shape)
        print("  shape v2: ", vertices_v2.shape)

    # Compare `segments.json` (segIndices)
    seg_indices_file_v1 = data_path_v1 / "segments.json"
    seg_indices_file_v2 = data_path_v2 / "segments.json"
    seg_indices_v1 = load_seg_indices(seg_indices_file_v1)
    seg_indices_v2 = load_seg_indices(seg_indices_file_v2)
    print("  indices shape", seg_indices_v2.shape)

    if seg_indices_v1 is None or seg_indices_v2 is None:
        print("  segments.json: File not found")
    elif np.array_equal(seg_indices_v1, seg_indices_v2):
        print("  segments.json: segIndices match")
    else:
        print("  segments.json: segIndices do not match")

    # Compare `segments_anno.json` (labels and segments)
    seg_anno_file_v1 = data_path_v1 / "segments_anno.json"
    seg_anno_file_v2 = data_path_v2 / "segments_anno.json"
    labels_v1, segments_v1 = load_seg_annotations(seg_anno_file_v1)
    labels_v2, segments_v2 = load_seg_annotations(seg_anno_file_v2)

    if labels_v1 is None or labels_v2 is None:
        print("  segments_anno.json: File not found")
    else:
        if labels_v1 == labels_v2:
            print("  segments_anno.json: Labels match")
        else:
            print("  segments_anno.json: Labels do not match")

        if segments_v1 == segments_v2:
            print("  segments_anno.json: Segments match")
        else:
            print("  segments_anno.json: Segments do not match")
