import cv2
from scripts.clip_embedding import embed_image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.transform import Rotation as R
import numpy as np

TRESHOLD = .7

# decides if VLM should be reprompted based on scene difference
def should_reprompt(rgb_frame, pos):

    # return False

    SEMANTIC_WEIGHT = 0.2
    GRAY_WEIGHT = 0.2
    ROT_WEIGHT = 0.3
    POS_WEIGHT = 0.2
    TIME_WEIGHT = 0.3
    
    # Unpack 
    prev_rgb_frame, cur_rgb_frame = rgb_frame
    prev_pos, cur_pos = pos

    # get semantic frames
    prev_semantic_frame, cur_semantic_frame = embed_image(prev_rgb_frame), embed_image(cur_rgb_frame)

    # get gray frames
    prev_gray = cv2.cvtColor(prev_rgb_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(cur_rgb_frame, cv2.COLOR_BGR2GRAY)

    # get gray score
    gray_diff = cv2.absdiff(gray_curr, prev_gray)
    gray_delta = gray_diff.mean() / 255.0
    gray_delta *= 5

    # cosine similarity score 
    semantic_delta = 1 - cosine_similarity(
        [prev_semantic_frame],
        [cur_semantic_frame]
    )[0, 0]
    semantic_delta *= 3

    # unpack positonal data
    time_delta = (
        cur_pos["timestamp"] -
        prev_pos["timestamp"]
    )

    translation_delta = np.linalg.norm([
        cur_pos["tx"] - prev_pos["tx"],
        cur_pos["ty"] - prev_pos["ty"],
        cur_pos["tz"] - prev_pos["tz"],
    ])
    translation_delta *= 100

    prev_rot = R.from_quat([
        prev_pos["qx"],
        prev_pos["qy"],
        prev_pos["qz"],
        prev_pos["qw"],
    ])

    cur_rot = R.from_quat([
        cur_pos["qx"],
        cur_pos["qy"],
        cur_pos["qz"],
        cur_pos["qw"],
    ])

    rotation_delta = (
        prev_rot.inv() * cur_rot
    ).magnitude()
    rotation_delta *= 1000

    # print(f"DIFFERENCE SCORE: {score}")

    # Final Probability
    prob = (
        GRAY_WEIGHT * gray_delta +
        SEMANTIC_WEIGHT * semantic_delta +
        TIME_WEIGHT * time_delta +
        POS_WEIGHT * translation_delta +
        ROT_WEIGHT * rotation_delta
    )

    if prob > TRESHOLD:
        run_gpt = True
    else:
        run_gpt = False

    # run_gpt = False # override to false for debugging

    # print("\n========== REPROMPT DEBUG ==========")
    # print(f"GRAY:      {GRAY_WEIGHT * gray_delta:.4f}")
    # print(f"SEMANTIC:  {SEMANTIC_WEIGHT * semantic_delta:.4f}")
    # print(f"TIME:      {TIME_WEIGHT * time_delta:.4f}")
    # print(f"POSITION:  {POS_WEIGHT * translation_delta:.4f}")
    # print(f"ROTATION:  {ROT_WEIGHT * rotation_delta:.4f}")
    # print(f"ROTATION DELTA:  {rotation_delta:.4f}")
    # print("PREV ROT:", [prev_pos[k] for k in ("qx","qy","qz","qw")])
    # print("CURR ROT:", [cur_pos[k] for k in ("qx","qy","qz","qw")])
    # print("-" * 35)
    print(f"FINAL:     {prob:.4f}")
    # print(f"RUN GPT:   {run_gpt}")

    return run_gpt
