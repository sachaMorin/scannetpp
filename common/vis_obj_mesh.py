import open3d as o3d
import json
import numpy as np
from pathlib import Path
import argparse


class InteractiveVisualizer:
    def __init__(self, mesh_file, seg_indices_file, seg_anno_file):
        self.mesh, self.seg_indices, self.seg_anno = self.load_mesh_and_segments(
            mesh_file, seg_indices_file, seg_anno_file
        )
        self.original_mesh = self.mesh  # Keep a copy for reset
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()

        self.print_valid_object_ids()

    def load_mesh_and_segments(self, mesh_file, seg_indices_file, seg_anno_file):
        mesh_file = str(mesh_file)
        seg_indices_file = str(seg_indices_file)
        seg_anno_file = str(seg_anno_file)

        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()

        with open(seg_indices_file, "r") as f:
            seg_indices = np.array(json.load(f)["segIndices"])

        with open(seg_anno_file, "r") as f:
            seg_anno = json.load(f)["segGroups"]

        return mesh, seg_indices, seg_anno

    def print_valid_object_ids(self):
        print("---- Valid Object IDs and their Labels ----")
        for group in self.seg_anno:
            object_id = group["objectId"]
            label = group["label"]
            print(f"Object ID: {object_id}, Label: {label}")
        print("------------------------------------------")

    def filter_mesh_by_object_id(self, object_id):
        # Collect all segment IDs corresponding to the object_id
        target_segments = set()
        for group in self.seg_anno:
            if group["objectId"] == object_id:
                target_segments.update(
                    group["segments"]
                )  # Add all segments from the list

        if not target_segments:
            print(f"No segments found for Object ID {object_id}.")
            return None

        target_vertices = np.isin(self.seg_indices, list(target_segments)).nonzero()[0]

        # Create a new mesh with only the filtered vertices
        filtered_mesh = self.mesh.select_by_index(target_vertices)
        return filtered_mesh

    def key_callback_filter(self, vis):
        """Callback to filter and display a specific object by ID."""

        object_id = input("Enter the Object ID to visualize")
        try:
            object_id = int(object_id)

            filtered_mesh = self.filter_mesh_by_object_id(object_id)
            if filtered_mesh:
                vis.clear_geometries()
                vis.add_geometry(filtered_mesh)
                print(f"Displaying Object ID {object_id}.")
        except ValueError:
            print("Invalid input. Please enter a valid Object ID.")

    def key_callback_reset(self, vis):
        """Callback to reset the view to the original mesh."""

        vis.clear_geometries()
        vis.add_geometry(self.original_mesh)
        print("Reset to original view.")

    def run_visualizer(self):
        # Initialize the visualizer
        self.visualizer.create_window("Interactive Visualizer")
        self.visualizer.add_geometry(self.mesh)

        # Register key callbacks
        self.visualizer.register_key_callback(ord("F"), self.key_callback_filter)
        self.visualizer.register_key_callback(ord("R"), self.key_callback_reset)

        # Run the visualizer
        print("Press 'F' to filter by Object ID.")
        print("Press 'R' to reset the view.")
        self.visualizer.run()
        self.visualizer.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Mesh Visualizer")
    parser.add_argument(
        "--scene_id", required=True, type=str, help="The ID of the scene to visualize"
    )
    args = parser.parse_args()

    data_path = Path("/data/concept-graphs/scannetpp/data") / args.scene_id / "scans"
    mesh_file = data_path / "mesh_aligned_0.05.ply"
    seg_indices_file = data_path / "segments.json"
    seg_anno_file = data_path / "segments_anno.json"

    vis = InteractiveVisualizer(mesh_file, seg_indices_file, seg_anno_file)
    vis.run_visualizer()
