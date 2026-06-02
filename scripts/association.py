from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np

def association(new_data, all_data):
    node_id, embedding, segmented, depth, pose = new_data
    node_ids, embedding_matrix, segmented_objs, depth_mem, poses = all_data

    if len(embedding_matrix) == 0:
        return None, None, None

    CLIP_WEIGHT = 0.5
    RGB_WEIGHT = 0.2
    DEPTH_WEIGHT = 0.3
    POSE_WEIGHT = 0.2

    # Cosine Sims
    cosine_sims = cosine_similarity([embedding], embedding_matrix)[0]

    # Depth Score
    curr_depth_mean = np.median(
        depth[depth > 0]
    )
    old_depth_means = np.array([
        np.median(d[d > 0])
        for d in depth_mem
    ])
    depth_diffs = np.abs(
        old_depth_means -
        curr_depth_mean
    )

    depth_scores = np.exp(-depth_diffs)

    # Pose Score
    poses_np = np.array([
        [p["tx"], p["ty"], p["tz"]]
        for p in poses
    ])
    curr_pose = np.array([
        pose["tx"],
        pose["ty"],
        pose["tz"]
    ])
    dists = np.linalg.norm(
        poses_np - curr_pose,
        axis=1
    )
    pose_scores = np.exp(-dists)

    # Final Probability
    probabilities = (
        CLIP_WEIGHT * cosine_sims +
        DEPTH_WEIGHT * depth_scores +
        POSE_WEIGHT * pose_scores
    )

    best_idx = np.argmax(probabilities)

    best_prob = probabilities[best_idx]

    best_node_id = node_ids[best_idx]

    print("\n========== BEST MATCH DEBUG ==========")

    print(f"CURR NODE:  {node_id}")
    print(f"BEST NODE:  {best_node_id}")

    print(f"CLIP:  {cosine_sims[best_idx]:.4f}")
    print(f"DEPTH: {depth_scores[best_idx]:.4f}")
    print(f"POSE:  {pose_scores[best_idx]:.4f}")

    print(f"FINAL: {best_prob:.4f}")

    match_img = cv2.resize(
        segmented_objs[best_idx],
        (
            segmented.shape[1],
            segmented.shape[0]
        )
    )

    rgb_row = np.hstack([
        segmented,
        match_img
    ])

    # Depth Visualization
    depth_match = cv2.resize(
        depth_mem[best_idx],
        (
            depth.shape[1],
            depth.shape[0]
        )
    )

    depth_vis_1 = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_vis_2 = cv2.normalize(
        depth_match,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_row = np.hstack([
        depth_vis_1,
        depth_vis_2
    ])

    combined = np.vstack([
        rgb_row,
        depth_row
    ])

    cv2.imshow(
        "MATCHES",
        combined
    )

    cv2.waitKey(0)

    cv2.destroyWindow("MATCHES")

    return (
        best_prob,
        best_node_id,
        best_idx
    )


