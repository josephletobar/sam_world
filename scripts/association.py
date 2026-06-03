from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np

def association(new_data, all_data):
    node_id, embedding, segmented, pose = new_data
    node_ids, embedding_matrix, segmented_objs, poses = all_data

    if len(embedding_matrix) == 0:
        return None, None, None

    CLIP_WEIGHT = 0.6
    POSE_WEIGHT = 0.3
    COLOR_WEIGHT = 0.05
    SHAPE_WEIGHT = 0.05

    # Color Score
    curr_color = segmented[segmented.any(axis=2)].mean(axis=0)
    old_colors = np.array([
        obj[obj.any(axis=2)].mean(axis=0)
        for obj in segmented_objs
    ])
    color_scores = np.exp(
        -np.linalg.norm(old_colors - curr_color, axis=1) / 100
    )

    # Geometric Shape Score
    curr_mask = cv2.resize(
        (segmented.any(axis=2)).astype(np.uint8),
        (64, 64)
    )
    old_masks = [
        cv2.resize(
            (obj.any(axis=2)).astype(np.uint8),
            (64, 64)
        )
        for obj in segmented_objs
    ]
    shape_scores = np.array([
        1 - cv2.absdiff(curr_mask, old_mask).mean()
        for old_mask in old_masks
    ])

    # Cosine Sims
    cosine_sims = cosine_similarity([embedding], embedding_matrix)[0]

    # Pose Score
    poses_np = np.array(poses)
    curr_pose = np.array(pose)
    dists = np.linalg.norm(
        poses_np - curr_pose,
        axis=1
    )
    SIGMA = 2
    pose_scores = np.exp(-dists / SIGMA)

    # Final Probability
    probabilities = (
        CLIP_WEIGHT * cosine_sims +
        POSE_WEIGHT * pose_scores +
        COLOR_WEIGHT * color_scores +
        SHAPE_WEIGHT * shape_scores 
    )

    best_idx = np.argmax(probabilities)

    best_prob = probabilities[best_idx]

    best_node_id = node_ids[best_idx]

    # if best_prob > 0.8:
    print("\n========== BEST MATCH DEBUG ==========")
    print(f"CURR NODE:  {node_id}")
    print(f"BEST NODE:  {best_node_id}")
    print("CURRENT:", curr_pose)
    print("MATCHED:", poses_np[best_idx])
    print(f"CLIP:  {cosine_sims[best_idx]:.4f}")
    print(f"POSE:  {pose_scores[best_idx]:.4f}")
    print(f"COLOR:  {color_scores[best_idx]:.4f}")
    print(f"SHAPE:  {shape_scores[best_idx]:.4f}")
    print(f"FINAL: {best_prob:.4f}")

    # match_img = cv2.resize(
    #     segmented_objs[best_idx],
    #     (
    #         segmented.shape[1],
    #         segmented.shape[0]
    #     )
    # )
    # rgb_row = np.hstack([
    #     segmented,
    #     match_img
    # ])
    # cv2.imshow(
    #     "MATCHES",
    #     rgb_row
    # )
    # cv2.waitKey(0)
    # cv2.destroyWindow("MATCHES")

    return (
        best_prob,
        best_node_id,
        best_idx
    )


