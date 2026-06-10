from turtle import pos

import cv2
from utils.clip_embedding import embed_image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.transform import Rotation as R
import numpy as np

# decides if VLM should be reprompted based on scene difference

class SceneDifferenceDetector:

    def __init__(self, slam_frame, threshold=0.6, debug=False):
        self.threshold = threshold

        self.slam_frame = slam_frame


        self.debug = debug

        self.SEMANTIC_WEIGHT = 0.5
        self.RGB_WEIGHT = 0.4
        self.ROT_WEIGHT = 0.1
        self.POS_WEIGHT = 0.00
        self.TIME_WEIGHT = 0.0

        self.count = 0

        self.prev_frame = None
        self.prev_pos = None
    
        
    def _semantic_diff(self, frame):

        # Embed Frames
        prev_semantic_frame, cur_semantic_frame = embed_image(self.prev_frame), embed_image(frame)

        # Cosine Similarity Score 
        semantic_delta = 1 - cosine_similarity(
            [prev_semantic_frame],
            [cur_semantic_frame]
        )[0, 0]

        return semantic_delta
    
    def _rgb_diff(self, frame):
        rgb_diff = cv2.absdiff(frame, self.prev_frame)
        rgb_delta = rgb_diff.mean() / 255.0

        return rgb_delta
    
    def _time_diff(self, pos):
        time_delta = (
            pos["timestamp"] -
            self.prev_pos["timestamp"]
        )

        return time_delta
    
    def _translation_diff(self, pos):
        translation_delta = np.linalg.norm([
            pos["tx"] - self.prev_pos["tx"],
            pos["ty"] - self.prev_pos["ty"],
            pos["tz"] - self.prev_pos["tz"],
        ])

        return translation_delta

    def _rotation_diff(self, pos):
        prev_rot = R.from_quat([
            self.prev_pos["qx"],
            self.prev_pos["qy"],
            self.prev_pos["qz"],
            self.prev_pos["qw"],
        ])

        cur_rot = R.from_quat([
            pos["qx"],
            pos["qy"],
            pos["qz"],
            pos["qw"],
        ])

        rotation_delta = (
            prev_rot.inv() * cur_rot
        ).magnitude()

        return rotation_delta

    def should_reprompt(self, frame=None):
        if frame is None:
            if self.slam_frame is None:
                raise RuntimeError("SceneDifferenceDetector needs a current rgb frame")
            frame = self.slam_frame.rgb

        if self.slam_frame is None:
            raise RuntimeError("SceneDifferenceDetector needs a current pose")

        pos = self.slam_frame.pose

        if frame is None or pos is None:
            raise RuntimeError("SceneDifferenceDetector needs current rgb and pose")

        # First iteration TRUE
        if self.prev_frame is None or self.prev_pos is None:
            self.prev_frame = frame
            self.prev_pos = pos
            return True     
        
        # Get Feature Deltas
        semantic_delta = self._semantic_diff(frame)
        rgb_delta = self._rgb_diff(frame)
        time_delta = self._time_diff(pos)
        translation_delta = self._translation_diff(pos)
        rotation_delta = self._rotation_diff(pos)

        # Normalize
        rgb_norm = min(rgb_delta / 0.30, 1.0)
        semantic_norm = min(semantic_delta / 0.20, 1.0)
        pos_norm = min(translation_delta / 5.0, 1.0)
        rot_norm = min(rotation_delta / np.pi, 1.0)
        time_norm = min(time_delta / 30.0, 1.0)

        # Final Probability
        prob = (
            self.RGB_WEIGHT * rgb_norm +
            self.SEMANTIC_WEIGHT * semantic_norm +
            self.TIME_WEIGHT * time_norm +
            self.POS_WEIGHT * pos_norm +
            self.ROT_WEIGHT * rot_norm
        )

        if prob > self.threshold:
            self.prev_frame = frame
            self.prev_pos = pos
            diff = True
        else:
            diff = False


        if self.debug:
            print("\n========== RAW ==========")
            print(f"RGB_RAW:       {rgb_norm:.4f}")
            print(f"SEMANTIC_RAW:  {semantic_norm:.4f}")
            print(f"TIME_RAW:      {time_norm:.4f}")
            print(f"POS_RAW:       {pos_norm:.4f}")
            print(f"ROT_RAW:       {rot_norm:.4f}")

            print("\n======= WEIGHTED ========")
            print(f"RGB:       {self.RGB_WEIGHT * rgb_norm:.4f}")
            print(f"SEMANTIC:  {self.SEMANTIC_WEIGHT * semantic_norm:.4f}")
            print(f"TIME:      {self.TIME_WEIGHT * time_norm:.4f}")
            print(f"POSITION:  {self.POS_WEIGHT * pos_norm:.4f}")
            print(f"ROTATION:  {self.ROT_WEIGHT * rot_norm:.4f}")
            print("-" * 35)
            print(f"FINAL:     {prob:.4f}")
        print(f"FINAL SCENE DIFF:     {prob:.4f}")

        return diff
            
    


# if __name__ == "__main__":
#     import cv2
#     from pathlib import Path
#     from sam_world import SamWorld
    
#     # sam_instance = SamWorld(r"D:\forest_data")
#     sam_instance = SamWorld(r"D:\kab3_data")
    
#     while True:
#         # Get Frame Info
#         ret, frame, slam_dict = sam_instance.get_next_frame()
        
#         if not ret or frame is None:
#             continue

#         pose = slam_dict["pose"]
#         depth = slam_dict["depth"]

#         # First Iteration
#         if (
#             sam_instance.prev_gpt_call["frame"] is None
#             or
#             sam_instance.prev_gpt_call["position"] is None
#         ):
#             sam_instance.prev_gpt_call["frame"] = frame
#             sam_instance.prev_gpt_call["position"] = pose
#             diff = True


#         elif sam_instance.frame_count % sam_instance.CHANGE_STEP == 0:

#             prev_cur_frames =  sam_instance.prev_gpt_call["frame"], frame 
#             self.prev_pos = sam_instance.prev_gpt_call["position"], pose

#             sam_instance.diff, prob, scores = should_reprompt(prev_cur_frames, self.prev_pos)

#             if sam_instance.diff == True:
#                 debug_prev_frame = sam_instance.prev_gpt_call["frame"].copy()

#                 sam_instance.prev_gpt_call["frame"] = frame
#                 sam_instance.prev_gpt_call["position"] = pose

            

#         if sam_instance.diff:

#             prev_frame = debug_prev_frame.copy()
#             cur_frame = frame.copy()

#             h = max(prev_frame.shape[0], cur_frame.shape[0])

#             prev_frame = cv2.resize(
#                 prev_frame,
#                 (int(prev_frame.shape[1] * h / prev_frame.shape[0]), h)
#             )

#             cur_frame = cv2.resize(
#                 cur_frame,
#                 (int(cur_frame.shape[1] * h / cur_frame.shape[0]), h)
#             )

#             comparison = np.hstack([prev_frame, cur_frame])

#             gray_score, semantic_score, time_score, pos_score, rot_score = scores

#             lines = [
#                 f"GRAY      : {gray_score:.4f}",
#                 f"SEMANTIC  : {semantic_score:.4f}",
#                 f"TIME      : {time_score:.4f}",
#                 f"POSITION  : {pos_score:.4f}",
#                 f"ROTATION  : {rot_score:.4f}",
#                 "---------------------",
#                 f"FINAL     : {prob:.4f}",
#             ]

#             line_height = 30
#             padding = 10

#             box_w = 350
#             box_h = len(lines) * line_height + 2 * padding

#             cv2.rectangle(
#                 comparison,
#                 (10, 10),
#                 (10 + box_w, 10 + box_h),
#                 (255, 255, 255),
#                 -1
#             )

#             y = 40
#             for line in lines:
#                 cv2.putText(
#                     comparison,
#                     line,
#                     (20, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 0, 0),
#                     2
#                 )
#                 y += line_height

#             cv2.imshow("CHANGE DETECTED", comparison)

#             while True:
#                 key = cv2.waitKey(0)

#                 if key == ord(" "):
#                     break

#             cv2.destroyWindow("CHANGE DETECTED")

#         cv2.imshow("Diff Video", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
