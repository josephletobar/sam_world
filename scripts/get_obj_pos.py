import numpy as np

# Intrinsic Camera Calibration
FX = 525.0  # focal length x
FY = 525.0  # focal length y
CX = 319.5  # optical center x
CY = 239.5  # optical center y


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

    depth_values = segmented_depth[:, :, 0]
    
    depth_values = depth_values[depth_values > 0]
    if len(depth_values) == 0:
        return None

    depth_value = depth_values.mean()

    # cv2.imshow("DEPTH", slam_dict["depth"])
    # cv2.waitKey(0)
    # cv2.destroyWindow("DEPTH")

    # use camera intrinsics
    local_x = (cx - CX) * depth_value / FX
    local_y = (cy - CY) * depth_value / FY
    local_z = depth_value

    world_x = pose["tx"] + local_x
    world_y = pose["ty"] + local_y
    world_z = pose["tz"] + local_z

    world_pos = (
        world_x,
        world_y,
        world_z
    )

    if any(v is None for v in world_pos):
        return None
    if any(np.isnan(v) for v in world_pos):
        return None

    return (world_pos, (cx, cy))