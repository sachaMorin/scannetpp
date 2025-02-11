from pathlib import Path
import hashlib
import json
import numpy as np
import open3d as o3d

# Define the paths and scene IDs
scene_ids = [
    "0a76e06478",
    "1f7cbbdde1",
    "49a82360aa",
    "8a35ef3cfe",
    "d918af9c5f",
    "0a7cc12c0e",
    "410c470782",
    "4c5c60fa76",
    "c0f5742640",
    "fd361ab85f",
]

data_path_v1_base = Path("/data/concept-graphs/scannetpp/data")
data_path_v2_base = Path("/data/concept-graphs/scannetpp_v2/data")

files_to_check = ["mesh_aligned_0.05.ply", "segments.json", "segments_anno.json"]


# Function to calculate the hash of a file
def calculate_hash(file_path, hash_algorithm="sha256"):
    hash_func = hashlib.new(hash_algorithm)
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except FileNotFoundError:
        return None


# Compare hashes for each scene ID and file
for scene_id in scene_ids:
    print(f"\nChecking scene_id: {scene_id}")
    # data_path_v1 = data_path_v1_base / scene_id / "scans"
    # data_path_v2 = data_path_v2_base / scene_id / "scans"

    # for file_name in files_to_check:
    #     file_v1 = data_path_v1 / file_name
    #     file_v2 = data_path_v2 / file_name

    #     hash_v1 = calculate_hash(file_v1)
    #     hash_v2 = calculate_hash(file_v2)

    #     if hash_v1 is None or hash_v2 is None:
    #         status = "File not found"
    #     elif hash_v1 == hash_v2:
    #         status = "Hashes match"
    #     else:
    #         status = "Hashes do not match"

    #     print(f"  {file_name}: {status}")

    poses_path_v1 = data_path_v1_base / scene_id / "iphone" / "colmap" / "images.txt"
    poses_path_v2 = data_path_v2_base / scene_id / "iphone" / "colmap" / "images.txt"

    hash_poses_v1 = calculate_hash(poses_path_v1)
    hash_poses_v2 = calculate_hash(poses_path_v2)

    if hash_poses_v1 is None or hash_poses_v2 is None:
        status = "File not found"
    elif hash_poses_v1 == hash_poses_v2:
        status = "Hashes match"
    else:
        status = "Hashes do not match"

    print(f"  images.txt: {status}")
