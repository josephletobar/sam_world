import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

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
        sam_model_path="models/sam3.pt",
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

        overrides = dict(
            conf=0.8,
            task="segment",
            mode="predict",
            model=sam_model_path,
            save=False,
        )
        self.sam_predictor = SAM3SemanticPredictor(overrides=overrides)

        torch.set_grad_enabled(False)

    def _prepare_frame(self, frame):
        if self.resize_to is None:
            return frame
        return cv2.resize(frame, self.resize_to)

    def initialize(self, frame, labels):
        frame = self._prepare_frame(frame)

        with torch.no_grad():
            self.sam_predictor.set_image(frame)
            results = self.sam_predictor(text=labels)

            if not len(results) or results[0].masks is None:
                raise ValueError("SAM3 did not find masks for the requested labels")

            result = results[0]
            sam_masks = result.masks.data.float()
            self.inference.set_all_labels(list(range(1, sam_masks.shape[0] + 1)))

            img_torch, _ = image_to_torch(frame, self.device)
            self.inference.step(img_torch, sam_masks)

        self.initialized = True
        self._cleanup_sam()

        return {
            "annotated": result.plot(),
            "masks": sam_masks.detach().cpu().numpy(),
        }

    def track(self, frame):
        if not self.initialized:
            raise RuntimeError("Call initialize(frame, labels) before track(frame)")

        frame = self._prepare_frame(frame)

        with torch.no_grad():
            img_torch, _ = image_to_torch(frame, self.device)
            masks = self.inference.step(img_torch)

        object_probs = masks[1:].detach().cpu().numpy()
        segmented_rgbs = self.get_segmented_rgbs(frame, object_probs)
        annotated = self.visualize(frame, object_probs)

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
        annotated = frame.copy()

        for object_idx, mask_prob in enumerate(object_probs):
            mask_prob = mask_prob.copy()
            mask_prob[mask_prob < min_prob] = 0.0
            mask_binary = mask_prob > threshold

            if not mask_binary.any():
                continue

            color = PALETTE[object_idx % len(PALETTE)]
            alpha = mask_prob * 0.6

            for c in range(3):
                annotated[:, :, c] = (
                    annotated[:, :, c] * (1 - alpha) +
                    color[c] * alpha
                ).astype(np.uint8)

        return annotated

    def _cleanup_sam(self):
        if (
            hasattr(self.sam_predictor, "predictor")
            and hasattr(self.sam_predictor.predictor, "features")
        ):
            self.sam_predictor.predictor.features = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def cleanup(self):
        self._cleanup_sam()


if __name__ == "__main__":
    labels = ["keyboard", "monitor"]

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

    init_result = tracker.initialize(first_img, labels)
    cv2.imshow("Segmentation", init_result["annotated"])

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