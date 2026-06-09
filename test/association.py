from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
from modules.ObjectPerception import WorldObject

SIM_THRESHOLD = 0.8



    # if best_prob > 0.8:
    # print("\n========== BEST MATCH DEBUG ==========")
    # print(f"CURR NODE:  {node_id}")
    # print(f"BEST NODE:  {best_node_id}")
    # print("CURRENT:", curr_pose)
    # print("MATCHED:", poses_np[best_idx])
    # print(f"CLIP:  {cosine_sims[best_idx]:.4f}")
    # print(f"POSE:  {pose_scores[best_idx]:.4f}")
    # print(f"COLOR:  {color_scores[best_idx]:.4f}")
    # print(f"SHAPE:  {shape_scores[best_idx]:.4f}")
    # print(f"FINAL: {best_prob:.4f}")

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

    


