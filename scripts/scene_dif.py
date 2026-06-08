import cv2
from scripts.clip_embedding import embed_image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.transform import Rotation as R
import numpy as np

TRESHOLD = .7

# decides if VLM should be reprompted based on scene difference
def should_reprompt(rgb_frame, pos):

    # return False

    SEMANTIC_WEIGHT = 0.4
    RGB_WEIGHT = 0.4
    ROT_WEIGHT = 0.1
    POS_WEIGHT = 0.1
    TIME_WEIGHT = 0.0
    
    # Unpack 
    prev_rgb_frame, cur_rgb_frame = rgb_frame
    prev_pos, cur_pos = pos

    # get semantic frames
    prev_semantic_frame, cur_semantic_frame = embed_image(prev_rgb_frame), embed_image(cur_rgb_frame)

    # get rgb frames
    rgb_diff = cv2.absdiff(cur_rgb_frame, prev_rgb_frame)
    rgb_delta = rgb_diff.mean() / 255.0

    # cosine similarity score 
    semantic_delta = 1 - cosine_similarity(
        [prev_semantic_frame],
        [cur_semantic_frame]
    )[0, 0]

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

    # normalize
    rgb_norm = min(rgb_delta / 0.30, 1.0)
    semantic_norm = min(semantic_delta / 0.20, 1.0)

    pos_norm = min(translation_delta / 5.0, 1.0)
    rot_norm = min(rotation_delta / np.pi, 1.0)
    time_norm = min(time_delta / 30.0, 1.0)

    # get pos rates
    translation_rate = translation_delta / time_delta
    rotation_rate = rotation_delta / time_delta

    # Final Probability
    prob = (
        RGB_WEIGHT * rgb_norm +
        SEMANTIC_WEIGHT * semantic_norm +
        TIME_WEIGHT * time_norm +
        POS_WEIGHT * pos_norm +
        ROT_WEIGHT * rot_norm
    )

    if prob > TRESHOLD:
        run_gpt = True
    else:
        run_gpt = False

    print("\n========== RAW ==========")
    print(f"RGB_RAW:       {rgb_norm:.4f}")
    print(f"SEMANTIC_RAW:  {semantic_norm:.4f}")
    print(f"TIME_RAW:      {time_norm:.4f}")
    print(f"POS_RAW:       {pos_norm:.4f}")
    print(f"ROT_RAW:       {rot_norm:.4f}")

    print("\n======= WEIGHTED ========")
    print(f"RGB:       {RGB_WEIGHT * rgb_norm:.4f}")
    print(f"SEMANTIC:  {SEMANTIC_WEIGHT * semantic_norm:.4f}")
    print(f"TIME:      {TIME_WEIGHT * time_norm:.4f}")
    print(f"POSITION:  {POS_WEIGHT * pos_norm:.4f}")
    print(f"ROTATION:  {ROT_WEIGHT * rot_norm:.4f}")
    print("-" * 35)
    print(f"FINAL:     {prob:.4f}")

    return (
        run_gpt,
        prob,
        (
            RGB_WEIGHT * rgb_norm,
            SEMANTIC_WEIGHT * semantic_norm,
            TIME_WEIGHT * time_norm,
            POS_WEIGHT * pos_norm,
            ROT_WEIGHT * rot_norm,
        )
    )

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from sam_world import SamWorld
    
    # sam_instance = SamWorld(r"D:\forest_data")
    sam_instance = SamWorld(r"D:\kab3_data")
    
    while True:
        # Get Frame Info
        ret, frame, slam_dict = sam_instance.get_next_frame()
        
        if not ret or frame is None:
            continue

        pose = slam_dict["pose"]
        depth = slam_dict["depth"]

        # First Iteration
        if (
            sam_instance.prev_gpt_call["frame"] is None
            or
            sam_instance.prev_gpt_call["position"] is None
        ):
            sam_instance.prev_gpt_call["frame"] = frame
            sam_instance.prev_gpt_call["position"] = pose
            run_gpt = True


        elif sam_instance.frame_count % sam_instance.CHANGE_STEP == 0:

            prev_cur_frames =  sam_instance.prev_gpt_call["frame"], frame 
            prev_cur_pos = sam_instance.prev_gpt_call["position"], pose

            sam_instance.run_gpt, prob, scores = should_reprompt(prev_cur_frames, prev_cur_pos)

            if sam_instance.run_gpt == True:
                debug_prev_frame = sam_instance.prev_gpt_call["frame"].copy()

                sam_instance.prev_gpt_call["frame"] = frame
                sam_instance.prev_gpt_call["position"] = pose

            

        if sam_instance.run_gpt:

            prev_frame = debug_prev_frame.copy()
            cur_frame = frame.copy()

            h = max(prev_frame.shape[0], cur_frame.shape[0])

            prev_frame = cv2.resize(
                prev_frame,
                (int(prev_frame.shape[1] * h / prev_frame.shape[0]), h)
            )

            cur_frame = cv2.resize(
                cur_frame,
                (int(cur_frame.shape[1] * h / cur_frame.shape[0]), h)
            )

            comparison = np.hstack([prev_frame, cur_frame])

            gray_score, semantic_score, time_score, pos_score, rot_score = scores

            lines = [
                f"GRAY      : {gray_score:.4f}",
                f"SEMANTIC  : {semantic_score:.4f}",
                f"TIME      : {time_score:.4f}",
                f"POSITION  : {pos_score:.4f}",
                f"ROTATION  : {rot_score:.4f}",
                "---------------------",
                f"FINAL     : {prob:.4f}",
            ]

            line_height = 30
            padding = 10

            box_w = 350
            box_h = len(lines) * line_height + 2 * padding

            cv2.rectangle(
                comparison,
                (10, 10),
                (10 + box_w, 10 + box_h),
                (255, 255, 255),
                -1
            )

            y = 40
            for line in lines:
                cv2.putText(
                    comparison,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )
                y += line_height

            cv2.imshow("CHANGE DETECTED", comparison)

            while True:
                key = cv2.waitKey(0)

                if key == ord(" "):
                    break

            cv2.destroyWindow("CHANGE DETECTED")

        cv2.imshow("Diff Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
