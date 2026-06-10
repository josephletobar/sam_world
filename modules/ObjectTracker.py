import sys
from pathlib import Path
from ultralytics.engine.results import Results

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor
from modules.ObjectPerception import WorldObject, ObjectPerception
from modules.Association import Association

sys.path.append(r"C:\Users\jletobar3\Projects\XMem")
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch
from model.network import XMem


CONFIG = {
    "mem_every": 20,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "max_mid_term_frames": 10,
    "min_mid_term_frames": 5,
    "max_long_term_elements": 1000,
    "num_prototypes": 128,
    "top_k": 30,
    "num_objects": 20,
}

PALETTE = np.array([
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
], dtype=np.uint8)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TrackObject:
    def __init__(
        self,
        xmem_model_path="models/XMem.pth",
        config=None,
        resize_to=(640, 480),
        device=None,
    ):
        self.device = device or get_device()
        self.config = dict(CONFIG if config is None else config)
        self.resize_to = resize_to
        self.initialized = False

        network = XMem(
            config=self.config,
            model_path=xmem_model_path
        ).to(self.device).eval()

        self.inference = InferenceCore(
            network=network,
            config=self.config
        )

        self.track_map = {}

        self.counter = 0

        torch.set_grad_enabled(False)

    def _prepare_frame(self, frame):
        if self.resize_to is None:
            return frame
        return cv2.resize(frame, self.resize_to)

    def _resize_masks(self, masks, size):
        return np.stack([
            cv2.resize(
                mask.astype(np.float32),
                size,
                interpolation=cv2.INTER_NEAREST
            )
            for mask in masks
        ])

    # Expects WorldObject Data Class
    def initialize(self, frame, objects: list[WorldObject]):
        frame = self._prepare_frame(frame)

        self.track_map = {}

        for track_id, obj in enumerate(objects, start=1):
            self.track_map[track_id] = obj.node_id

        masks = np.stack([obj.sam_mask.astype(np.float32) for obj in objects])

        if self.resize_to is not None:
            masks = self._resize_masks(masks, self.resize_to)

        masks = torch.tensor(
            masks,
            dtype=torch.float32,
            device=self.device
        )

        self.inference.set_all_labels(
            list(range(1, masks.shape[0] + 1))
        )

        with torch.no_grad():
            img_torch, _ = image_to_torch(frame, self.device)
            self.inference.step(img_torch, masks)

        self.initialized = True

    def track(self, frame):
        if not self.initialized:
            raise RuntimeError("Call initialize(frame, labels) before track(frame)")

        original_frame = frame
        frame = self._prepare_frame(frame)

        with torch.no_grad():
            img_torch, _ = image_to_torch(frame, self.device)
            masks = self.inference.step(img_torch)

        object_probs = masks[1:].detach().cpu().numpy()

        for i, mask_prob in enumerate(object_probs):

            confidence = mask_prob.max()

            if confidence < 0.7:
                object_probs[i] *= 0

        if frame.shape[:2] != original_frame.shape[:2]:
            original_size = (original_frame.shape[1], original_frame.shape[0])
            object_probs = self._resize_masks(object_probs, original_size)

        segmented_rgbs = self.get_segmented_rgbs(original_frame, object_probs)
        annotated = self.visualize(original_frame, object_probs)

        return {
            "annotated": annotated,
            "masks": object_probs,
            "segmented_rgbs": segmented_rgbs,
        }

    def get_segmented_rgbs(self, frame, object_probs, threshold=0.5):
        segmented_rgbs = []

        for mask_prob in object_probs:
            mask_binary = mask_prob > threshold

            if not mask_binary.any():
                segmented_rgbs.append(None)
                continue

            segmented_rgbs.append(frame * mask_binary[..., None])

        return segmented_rgbs

    def visualize(self, frame, object_probs, threshold=0.5, min_prob=0.9):

        print(self.track_map)
        print(object_probs.shape)

        annotated = frame.copy()

        labels_to_draw = []

        for object_idx, mask_prob in enumerate(object_probs):
            mask_prob = mask_prob.copy()
            mask_prob[mask_prob < min_prob] = 0.0
            mask_binary = mask_prob > threshold

            if not mask_binary.any():
                continue

            ys, xs = np.where(mask_binary)

            track_id = object_idx + 1
            node_id = self.track_map[track_id]

            labels_to_draw.append(
                (node_id, int(xs.mean()), int(ys.min()) - 10)
            )

            color = PALETTE[object_idx % len(PALETTE)]
            alpha = mask_prob * 0.6

            for c in range(3):
                annotated[:, :, c] = (
                    annotated[:, :, c] * (1 - alpha) +
                    color[c] * alpha
                ).astype(np.uint8)

        for node_id, x, y in labels_to_draw:
            cv2.putText(
                annotated,
                node_id,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated

    def cleanup(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    sam_model_path="models/sam3.pt",

    image_dir = Path(
        r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg1_xyz (1)\rgbd_dataset_freiburg1_xyz\rgb"
    )
    image_paths = [
        p for p in sorted(image_dir.glob("*"))
        if not p.name.startswith("._")
    ]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    tracker = TrackObject()
    

    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise ValueError(f"Could not read first image: {image_paths[0]}")
    
    slam_dict = {
        "rgb": first_img,
        "depth": np.ones(first_img.shape[:2], dtype=np.float32),
        "pose": {
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }
    
    object_perception = ObjectPerception()
    association = Association([], None)

    objects, annotated = object_perception.get_objects(first_img, ["keyboard", "monitor"], slam_dict)
    for obj in objects:
        association.update(obj)

    print(len(objects))

    tracker.initialize(first_img, objects)
    cv2.imshow("Segmentation", annotated)

    print("Press SPACE to start XMem tracking...")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 32:
            break

    for frame_idx, img_path in enumerate(image_paths[1:], start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        track_result = tracker.track(img)
        print(
            f"Frame {frame_idx} - "
            f"tracked masks: {track_result['masks'].shape}"
        )

        cv2.imshow("Segmentation", track_result["annotated"])

        if frame_idx % 10 == 0:
            tracker.cleanup()

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    tracker.cleanup()
