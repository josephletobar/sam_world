import cv2
import numpy as np
from dataclasses import dataclass
from ultralytics.models.sam import SAM3SemanticPredictor

from utils import association
from utils.get_obj_pos import get_pos
from utils.clip_embedding import embed_image, embed_text

@dataclass
class WorldObject:
    label: str
    confidence: float

    world_pos: tuple[float, float, float]
    image_pos: tuple[int, int]

    segmented_rgb: np.ndarray

    img_embedding: np.ndarray
    txt_embedding: np.ndarray

    node_id: str = None

class ObjectPerception:
    def __init__(self):

        overrides = dict(
            conf=0.8,
            task="segment",
            mode="predict",
            model="models/sam3.pt",
            save=False,
        )

        self.sam_predictor = SAM3SemanticPredictor(overrides=overrides)

        self.first_frame = True

    def get_objects(self, frame, labels, slam_dict):

        objects = []

        pose = slam_dict["pose"]
        depth = slam_dict["depth"]

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
            mask = mask.cpu().numpy()

            # Find objects pixel-wise position
            ys, xs = np.where(mask > 0.5)

            object_pose = get_pos(mask, depth, pose)
            if object_pose is None: continue

            world_pos, image_pos = object_pose

            segmented_rgb = frame.copy()
            segmented_rgb[mask == 0] = 0
            segmented_rgb = segmented_rgb[
                ys.min():ys.max(),
                xs.min():xs.max()
            ]

            # embed the object
            img_embedding = embed_image(segmented_rgb)
            txt_embedding = embed_text(label)

            objects.append(
                WorldObject(
                    label=label,
                    confidence=confidence,
                    world_pos=world_pos,
                    image_pos=image_pos,
                    segmented_rgb=segmented_rgb,
                    img_embedding=img_embedding,
                    txt_embedding=txt_embedding,
                )
            )

            return objects, annotated

            # # Object Association
            # new_data = (label, img_embedding, segmented_rgb, world_pos if slam_dict else None)
            # all_data = (self.labels, self.embedding_matrix, self.segmented_rgbs, self.world_poses if slam_dict else None)

            # if not self.first_frame:

            #     best_prob, best_id, best_idx = association(new_data, all_data)

            #     if best_prob < self.SIM_THRESHOLD:

            #         count = self.label_counts.get(label, 0)
            #         node_id = f"{label}_{count}"
            #         self.label_counts[label] = count + 1

            #         objects.append(
            #             WorldObject(
            #                 label=label,
            #                 confidence=confidence,
            #                 world_pos=world_pos,
            #                 image_pos=image_pos,
            #                 segmented_rgb=segmented_rgb,
            #                 img_embedding=img_embedding,
            #                 txt_embedding=txt_embedding,
            #                 node_id=node_id
            #             )
            #         )

            #     else: # Existing Node
            #         # Merge Nodes (skip for now)

            #         # # debug to visualize the matched objects
            #         # print(best_prob)
            #         # cv2.imshow("match1", segmented)
            #         # cv2.imshow("match2", self.segmented_rgbs[best_idx])
            #         # cv2.waitKey(0)
            #         # cv2.destroyAllWindows()
            #         pass

            # # First Node
            # else:
            #     self.first_frame = False
            #     count = self.label_counts.get(label, 0)
            #     node_id = f"{label}_{count}"
            #     self.label_counts[label] = count + 1

            #     objects.append(
            #             WorldObject(
            #                 label=label,
            #                 confidence=confidence,
            #                 world_pos=world_pos,
            #                 image_pos=image_pos,
            #                 segmented_rgb=segmented_rgb,
            #                 img_embedding=img_embedding,
            #                 txt_embedding=txt_embedding,
            #                 node_id=node_id
            #             )
            #         )

        return objects

            