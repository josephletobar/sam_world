import numpy as np
import cv2
from utils.get_obj_pos import get_pos
from utils.clip_embedding import embed_image, embed_text
from dataclasses import dataclass

@dataclass
class WorldObject:
    label: str
    confidence: float

    sam_mask: np.ndarray

    world_pos: tuple[float, float, float]
    image_pos: tuple[int, int]
    local_pos: np.ndarray
    translation: np.ndarray
    median_depth: float
    depth_stats: dict

    segmented_rgb: np.ndarray
    segmented_depth: np.ndarray

    img_embedding: np.ndarray
    txt_embedding: np.ndarray

    node_id: str = None
    first_seen: int = None
    last_seen: int = None

def create_object(frame, mask, label, depth, pose, confidence):

    if any(x is None for x in (frame, mask, label, depth, pose, confidence)):
        return None

    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()  
    # mask = cv2.resize(
    #     mask.astype(np.float32),
    #     (640, 480),
    #     interpolation=cv2.INTER_NEAREST
    # )
    mask = cv2.resize(mask.astype(np.float32), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Find objects pixel-wise position
    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0 or len(ys) == 0:
        return None

    object_pose = get_pos(mask, depth, pose)
    if object_pose is None: return

    (
        world_pos,
        image_pos,
        local_pos,
        translation,
        median_depth,
        depth_stats,
    ) = object_pose

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    if x2 <= x1 or y2 <= y1:
        return None

    segmented_rgb = frame.copy()
    segmented_rgb[mask == 0] = 0
    segmented_rgb = segmented_rgb[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    segmented_depth = depth.copy()
    segmented_depth[mask == 0] = 0
    segmented_depth = segmented_depth[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    if segmented_rgb.size == 0 or 0 in segmented_rgb.shape[:2]:
        return None

    # embed the object
    img_embedding = embed_image(segmented_rgb)
    txt_embedding = embed_text(label)

    return WorldObject(
        label=label,
        confidence=confidence,
        sam_mask=mask,
        world_pos=world_pos,
        image_pos=image_pos,
        local_pos=local_pos,
        translation=translation,
        median_depth=median_depth,
        depth_stats=depth_stats,
        segmented_rgb=segmented_rgb,
        segmented_depth=segmented_depth,
        img_embedding=img_embedding,
        txt_embedding=txt_embedding,
    )
