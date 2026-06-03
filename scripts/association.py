from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np

def association(new_data, all_data):
    node_id, embedding, segmented, pose = new_data
    node_ids, embedding_matrix, segmented_objs, poses = all_data

    if len(embedding_matrix) == 0:
        return None, None, None

    CLIP_WEIGHT = 0.7
    POSE_WEIGHT = 0.4

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
        POSE_WEIGHT * pose_scores
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


