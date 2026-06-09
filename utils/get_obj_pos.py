import numpy as np
from scipy.spatial.transform import Rotation as R

# Intrinsic Camera Calibration
FX = 525.0  # focal length x
FY = 525.0  # focal length y
CX = 319.5  # optical center x
CY = 239.5  # optical center y

DEPTH_SCALE = 5000.0

def get_pos(mask, depth, pose):    
    segmented_depth = depth.copy()
    segmented_depth[mask == 0] = 0

    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    cx = xs.mean()
    cy = ys.mean()
    dists = (xs - cx)**2 + (ys - cy)**2
    best_idx = np.argmin(dists)
    cx = xs[best_idx]
    cy = ys[best_idx]

    depth_values = segmented_depth

    depth_values = depth_values[depth_values > 0]
    if len(depth_values) == 0:
        return None

    depth_value = depth_values.mean()
    depth_value /= DEPTH_SCALE

    # print("-" * 40)
    # print(depth_value)
    # print("-" * 40)

    # cv2.imshow("DEPTH", slam_dict["depth"])
    # cv2.waitKey(0)
    # cv2.destroyWindow("DEPTH")

    # use camera intrinsics
    local_x = (cx - CX) * depth_value / FX
    local_y = (cy - CY) * depth_value / FY
    local_z = depth_value

    local_pos = np.array([
        local_x,
        local_y,
        local_z
    ])

    rotation = R.from_quat([
        pose["qx"],
        pose["qy"],
        pose["qz"],
        pose["qw"],
    ])

    translation = np.array([
        pose["tx"],
        pose["ty"],
        pose["tz"],
    ])

    world_pos = rotation.inv().apply(local_pos) + translation

    if any(v is None for v in world_pos):
        return None
    if any(np.isnan(v) for v in world_pos):
        return None

    return (world_pos, (cx, cy))