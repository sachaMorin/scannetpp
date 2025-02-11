import os
import json


def count_images_in_scenes(base_path: str):
    total_images = 0  # To keep track of the total number of images across all scenes
    total_background_objects = 0  # To keep track of background objects count

    # List of labels considered as background objects
    background_labels = ["background", "wall", "floor", "ceiling", "split", "remove"]

    # Loop through each scene in the base directory
    for scene_id in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_id, "iphone", "render_crops_sam2")
        metadata_path = os.path.join(
            base_path, scene_id, "iphone", "crops_data", "metadata"
        )

        if not os.path.isdir(scene_path):
            print(
                f"Scene ID: {scene_id} - 'render_crops_sam2' directory does not exist."
            )
            continue  # Skip if the directory does not exist

        # Count the number of image files in the scene path
        num_images = len(
            [
                f
                for f in os.listdir(scene_path)
                if os.path.isfile(os.path.join(scene_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]
        )

        # If metadata folder does not exist, assume no metadata to process
        if not os.path.isdir(metadata_path):
            print(f"Scene ID: {scene_id} - 'metadata' directory does not exist.")
            continue

        # Count background objects based on metadata
        num_background_objects = 0
        for metadata_file in os.listdir(metadata_path):
            if metadata_file.endswith("_metadata.json"):
                metadata_file_path = os.path.join(metadata_path, metadata_file)

                with open(metadata_file_path, "r") as file:
                    metadata = json.load(file)
                    label = metadata.get("label", "").lower()

                    if label in background_labels:
                        num_background_objects += 1

        # Calculate the number of objects excluding background objects
        num_objects_without_background = num_images - num_background_objects

        print(
            f"Scene ID: {scene_id}, Total objects: {num_images}, Background objects: {num_background_objects}, Objects without background: {num_objects_without_background}"
        )

        # Accumulate the counts for the total
        total_images += num_images
        total_background_objects += num_background_objects

    # Print the total number of images and background objects across all scenes
    total_objects_without_background = total_images - total_background_objects
    print(f"Total number of objects across all scenes: {total_images}")
    print(f"Total number of background objects: {total_background_objects}")
    print(
        f"Total number of objects without background: {total_objects_without_background}"
    )


# Path to the data directory
data_dir = "/data/concept-graphs/scannetpp/data"
count_images_in_scenes(data_dir)
