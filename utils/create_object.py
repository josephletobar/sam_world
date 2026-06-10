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

    segmented_rgb: np.ndarray

    img_embedding: np.ndarray
    txt_embedding: np.ndarray

    node_id: str = None

def create_object(frame, mask, label, depth, pose, confidence):

    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()  
    mask = cv2.resize(
        mask.astype(np.float32),
        (640, 480),
        interpolation=cv2.INTER_NEAREST
    )

    # Find objects pixel-wise position
    ys, xs = np.where(mask > 0.5)

    object_pose = get_pos(mask, depth, pose)
    if object_pose is None: return

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

    return WorldObject(
        label=label,
        confidence=confidence,
        sam_mask=mask,
        world_pos=world_pos,
        image_pos=image_pos,
        segmented_rgb=segmented_rgb,
        img_embedding=img_embedding,
        txt_embedding=txt_embedding,
    )