import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

from utils.get_obj_pos import get_pos
from utils.clip_embedding import embed_image, embed_text
from utils.create_object import create_object


class ObjectPerception:
    def __init__(self, slam_frame=None):
        self.slam_frame = slam_frame

        overrides = dict(
            conf=0.8,
            task="segment",
            mode="predict",
            model="models/sam3.pt",
            save=False,
        )

        self.sam_predictor = SAM3SemanticPredictor(overrides=overrides)

        self.first_frame = True

    def get_objects(self, *args, frame=None, labels=None, slam_dict=None):
        if len(args) >= 2:
            frame, labels = args[:2]
            if len(args) >= 3:
                slam_dict = args[2]
        elif len(args) == 1:
            labels = args[0]

        objects = []

        if frame is None:
            if self.slam_frame is None:
                raise RuntimeError("ObjectPerception needs a current rgb frame")
            frame = self.slam_frame.rgb

        if slam_dict is not None:
            pose = slam_dict["pose"]
            depth = slam_dict["depth"]
        else:
            if self.slam_frame is None:
                raise RuntimeError("ObjectPerception needs current depth and pose")
            pose = self.slam_frame.pose
            depth = self.slam_frame.depth

        if frame is None or depth is None or pose is None:
            raise RuntimeError("ObjectPerception needs current rgb, depth, and pose")

        frame = cv2.resize(frame, (640, 480))

        if depth.shape[:2] != frame.shape[:2]:
            depth = cv2.resize(
                depth,
                (640, 480),
                interpolation=cv2.INTER_NEAREST
            )

        results = self.sam_predictor(
            frame,
            text=labels,
            save=False,
            verbose=False
        )
        result = results[0]
        annotated = result.plot()

        if result.boxes is None or len(result.boxes) == 0:
            print("No detections")
            return [], frame

        # Go through each detected object
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            confidence = float(box.conf[0])

            mask = result.masks.data[i]

            ret_obj = create_object(frame, mask, label, depth, pose, confidence)

            if ret_obj is not None:
                objects.append(ret_obj)

        return objects, annotated

          
            
