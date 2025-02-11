from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the paths and scene IDs
scene_ids = [
    "8a35ef3cfe",
    "0a7cc12c0e",
    "4c5c60fa76",
]

data_path_v1_base = Path("/data/concept-graphs/scannetpp_openlex_v1/data")
data_path_v2_base = Path("/data/concept-graphs/scannetpp_openlex_v2/data")


def match_segments(segments_v1, obj_ids_v1, segments_v2, obj_ids_v2):
    """
    Matches segments from v1 to v2 in two stages:
      1) Perfect match (same segments ignoring ordering)
      2) IoU > 0.5 match
    Any remaining v1_obj_ids are assigned a match of -1.
    Returns a list of tuples (v1_obj_id, v2_obj_id).
    """

    def lists_equal(seg_a, seg_b):
        return sorted(seg_a) == sorted(seg_b)

    def iou(seg_a, seg_b):
        s1, s2 = set(seg_a), set(seg_b)
        inter = len(s1 & s2)
        uni = len(s1 | s2)
        return inter / uni if uni else 0

    dict_v1 = {obj_id: seg for obj_id, seg in zip(obj_ids_v1, segments_v1)}
    dict_v2 = {obj_id: seg for obj_id, seg in zip(obj_ids_v2, segments_v2)}

    unmatched_v1 = set(dict_v1.keys())
    unmatched_v2 = set(dict_v2.keys())
    matched_pairs = []

    # Pass 1: Perfect match
    for oid1 in sorted(obj_ids_v1):
        if oid1 not in unmatched_v1:
            continue
        seg1 = dict_v1[oid1]
        for oid2 in sorted(obj_ids_v2):
            if oid2 not in unmatched_v2:
                continue
            seg2 = dict_v2[oid2]
            if lists_equal(seg1, seg2):
                matched_pairs.append((oid1, oid2))
                unmatched_v1.remove(oid1)
                unmatched_v2.remove(oid2)
                break

    # Pass 2: IoU > 0.5
    for oid1 in sorted(list(unmatched_v1)):
        seg1 = dict_v1[oid1]
        found_match = False
        for oid2 in sorted(list(unmatched_v2)):
            seg2 = dict_v2[oid2]
            if iou(seg1, seg2) > 0.5:
                matched_pairs.append((oid1, oid2))
                unmatched_v1.remove(oid1)
                unmatched_v2.remove(oid2)
                found_match = True
                break

    # Pass 3: If no match found, assign -1
    for oid1 in sorted(list(unmatched_v1)):
        matched_pairs.append((oid1, -1))

    return matched_pairs


def compare_segments_bruteforce_and_print_iou_heatmap(
    scene_id, segments_v1, obj_ids_v1, segments_v2, obj_ids_v2
):
    def lists_equal(seg_a, seg_b):
        # "Perfect match" for segments can be defined different ways;
        # here we check if sorted sublists match exactly.
        return sorted(seg_a) == sorted(seg_b)

    def iou(seg_a, seg_b):
        s1, s2 = set(seg_a), set(seg_b)
        inter = len(s1 & s2)
        uni = len(s1 | s2)
        return inter / uni if uni else 0

    dict_v1 = {obj_id: seg for obj_id, seg in zip(obj_ids_v1, segments_v1)}
    dict_v2 = {obj_id: seg for obj_id, seg in zip(obj_ids_v2, segments_v2)}

    ids_v1_list = sorted(dict_v1.keys())
    ids_v2_list = sorted(dict_v2.keys())

    # Brute-force to remove objects with perfect match
    unmatched_v1 = set(ids_v1_list)
    unmatched_v2 = set(ids_v2_list)
    for oid1 in ids_v1_list:
        seg1 = dict_v1[oid1]
        for oid2 in ids_v2_list:
            seg2 = dict_v2[oid2]
            if lists_equal(seg1, seg2):
                unmatched_v1.discard(oid1)
                unmatched_v2.discard(oid2)
                break

    if not unmatched_v1 or not unmatched_v2:
        print(f"No unmatched objects in scene {scene_id}")
        return [], []

    # Build IoU matrix for unmatched only
    unmatched_v1_list = sorted(list(unmatched_v1))
    unmatched_v2_list = sorted(list(unmatched_v2))
    iou_matrix = np.zeros((len(unmatched_v1_list), len(unmatched_v2_list)))

    for i, oid1 in enumerate(unmatched_v1_list):
        for j, oid2 in enumerate(unmatched_v2_list):
            iou_matrix[i, j] = iou(dict_v1[oid1], dict_v2[oid2])

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        iou_matrix,
        xticklabels=unmatched_v2_list,
        yticklabels=unmatched_v1_list,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
    )
    plt.title(f"IoU Heatmap (Unmatched) - {scene_id}")
    plt.tight_layout()
    plt.savefig(f"iou_heatmap_unmatched_{scene_id}.png")
    plt.close()

    # Return unmatched info
    unmatched_list_v1 = [(oid, dict_v1[oid]) for oid in unmatched_v1_list]
    unmatched_list_v2 = [(oid, dict_v2[oid]) for oid in unmatched_v2_list]
    return unmatched_list_v1, unmatched_list_v2


def count_label_differences_counter(labels_v1, labels_v2):
    """
    Uses `Counter` to find how many labels differ in total between two lists.

    - For each label, compare how many times it appears in v1 vs v2.
    - The difference in counts for each label is how many are unmatched.
    - Summing over all labels yields total unmatched labels.
    """
    counter1 = Counter(labels_v1)
    counter2 = Counter(labels_v2)

    # Union of all unique labels in both lists
    all_labels = set(counter1.keys()) | set(counter2.keys())

    total_diff = 0
    for label in all_labels:
        total_diff += abs(counter1[label] - counter2[label])

    return total_diff


def count_differences_bruteforce(v1, v2, ignore_sublist_order=True):
    """
    Returns the total number of unmatched items when comparing two lists of lists (v1, v2).
    - If ignore_sublist_order is True, treats [1,2] as the same as [2,1].
    - Each sublist in v1 tries to find one matching sublist in v2 and removes it from v2's unmatched pool.
    - Anything in v1 that can't find a match is 'changed';
      plus anything still left in v2 is also 'changed'.
    """

    # Canonical form to ignore internal order if needed
    def canonical(sublist):
        return tuple(sorted(sublist)) if ignore_sublist_order else tuple(sublist)

    v2_unmatched = [(j, item) for j, item in enumerate(v2)]

    matched_pairs = []

    # Try to match each sublist in v1 with something in v2_unmatched
    for i, sub1 in enumerate(v1):
        c1 = canonical(sub1)
        match_index_in_v2 = -1  # index in v2_unmatched
        match_original_j = -1  # original index j in the v2 list

        # Search for a match in v2_unmatched
        for k, (j, sub2) in enumerate(v2_unmatched):
            if c1 == canonical(sub2):
                match_index_in_v2 = k
                match_original_j = j
                break

        # If found, remove from v2_unmatched and record the (i, j) pair
        if match_index_in_v2 != -1:
            del v2_unmatched[match_index_in_v2]
            matched_pairs.append((i, match_original_j))

    # matched_count is how many sublists we successfully paired
    matched_count = len(matched_pairs)

    # # of unmatched in v1 is everything we didn't match
    unmatched_v1 = len(v1) - matched_count
    # # of unmatched in v2 is everything left in v2_unmatched
    unmatched_v2 = len(v2_unmatched)
    diff_count = unmatched_v1 + unmatched_v2
    return diff_count


def compare_labels_by_obj_id(obj_ids_v1, labels_v1, obj_ids_v2, labels_v2):
    """
    Compare labels between two annotation versions by matching object IDs.
    - If an obj_id in v1 does not exist in v2, that's a difference.
    - If it exists in both but labels differ, that's a difference.
    - If an obj_id in v2 does not exist in v1, that's also a difference.

    Returns the total number of differences.
    """

    # Build a dict: {obj_id -> label}
    dict_v1 = {obj_id: label for obj_id, label in zip(obj_ids_v1, labels_v1)}
    dict_v2 = {obj_id: label for obj_id, label in zip(obj_ids_v2, labels_v2)}

    diff_count = 0

    # 1) Check each obj_id in v1
    for obj_id, label_v1 in dict_v1.items():
        if obj_id not in dict_v2:
            # obj_id is missing in v2
            diff_count += 1
        else:
            # obj_id in both: compare labels
            label_v2 = dict_v2[obj_id]
            if label_v1 != label_v2:
                diff_count += 1

    # 2) Check for any obj_id in v2 that doesn't exist in v1
    for obj_id in dict_v2.keys():
        if obj_id not in dict_v1:
            diff_count += 1

    return diff_count


def compare_segments_by_obj_id(obj_ids_v1, segments_v1, obj_ids_v2, segments_v2):
    """
    Compare segment annotations between two versions by matching object IDs.
    - If an obj_id in v1 does not exist in v2, difference += 1
    - If obj_id exists in both, compare their segment lists (as multisets).
    - If those multisets differ, difference += 1
    - If an obj_id in v2 does not exist in v1, difference += 1

    Returns the total number of differences.
    """

    # Build dictionary: {obj_id -> list_of_segments}
    dict_v1 = {obj_id: seg_list for obj_id, seg_list in zip(obj_ids_v1, segments_v1)}
    dict_v2 = {obj_id: seg_list for obj_id, seg_list in zip(obj_ids_v2, segments_v2)}

    diff_count = 0

    # 1) Check objects from v1
    for obj_id, seg_list_v1 in dict_v1.items():
        if obj_id not in dict_v2:
            diff_count += 1  # object missing in v2
        else:
            # Compare the segments for this object
            seg_list_v2 = dict_v2[obj_id]
            # Canonicalize -> compare as counters
            c1 = sorted(seg_list_v1)
            c2 = sorted(seg_list_v2)
            if c1 != c2:
                diff_count += 1

    # 2) Check objects that are in v2 but not in v1
    for obj_id in dict_v2.keys():
        if obj_id not in dict_v1:
            diff_count += 1

    return diff_count


# Function to load segment annotations (labels and segments)
def load_seg_annotations(seg_anno_file):
    try:
        with open(seg_anno_file, "r") as f:
            seg_anno = json.load(f)["segGroups"]
        obj_ids = [item["id"] for item in seg_anno]
        labels = [item["label"] for item in seg_anno]
        segments = [item["segments"] for item in seg_anno]
        return obj_ids, labels, segments
    except FileNotFoundError:
        return None, None, None


# Compare data for each scene ID
for scene_id in scene_ids:
    print(f"\nChecking scene_id: {scene_id}")
    data_path_v1 = data_path_v1_base / scene_id / "scans"
    data_path_v2 = data_path_v2_base / scene_id / "scans"

    # Load `segments_anno.json` data
    seg_anno_file_v1 = data_path_v1 / "segments_anno.json"
    seg_anno_file_v2 = data_path_v2 / "segments_anno.json"
    obj_ids_v1, labels_v1, segments_v1 = load_seg_annotations(seg_anno_file_v1)
    obj_ids_v2, labels_v2, segments_v2 = load_seg_annotations(seg_anno_file_v2)

    # Check if files are missing
    if labels_v1 is None or labels_v2 is None:
        print("  segments_anno.json: One of the files is missing.")
        continue

    # Count label differences
    # changed_labels_count = sum(1 for l1, l2 in zip(labels_v1, labels_v2) if l1 != l2)
    changed_labels_count = count_label_differences_counter(labels_v1, labels_v2)
    # changed_labels_count = compare_labels_by_obj_id(
    #     obj_ids_v1, labels_v1, obj_ids_v2, labels_v2
    # )
    print(f"  Labels changed for {changed_labels_count} objects.")

    # Count segment differences
    changed_segments_count = 0
    print(
        f"  Comparing segments for {len(segments_v1)} objects and {len(segments_v2)} objects."
    )
    # for seg1, seg2 in zip(segments_v1, segments_v2):
    #     if set(seg1) != set(seg2):  # Compare segment lists as sets to ignore order
    #         changed_segments_count += 1
    changed_segments_count = count_differences_bruteforce(segments_v1, segments_v2)
    # changed_segments_count = compare_segments_by_obj_id(
    #     obj_ids_v1, segments_v1, obj_ids_v2, segments_v2
    # )
    print(f"  Segments changed for {changed_segments_count} objects.")

    # Run match_segments
    matched_pairs = match_segments(segments_v1, obj_ids_v1, segments_v2, obj_ids_v2)
    print(f"  Matched pairs: {matched_pairs}")

    # convert to numpy arrays and save matched pairs
    matched_pairs_np = np.array(matched_pairs)
    np.save(f"matched_pairs_{scene_id}.npy", matched_pairs_np)
