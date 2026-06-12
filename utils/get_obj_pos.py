import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import camera_config

def get_pos(mask, depth, pose):    
    camera_config.require_camera_configured()

    mask = np.squeeze(mask)

    if depth.ndim > 2:
        depth = depth[:, :, 0]

    if mask.ndim > 2:
        if mask.shape[:2] == depth.shape[:2]:
            mask = mask.max(axis=2)
        else:
            mask = mask.max(axis=0)

    segmented_depth = depth.copy()
    segmented_depth[mask == 0] = 0

    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None

    depth_values = segmented_depth[ys, xs]
    valid = depth_values > 0

    if not np.any(valid):
        return None

    xs = xs[valid]
    ys = ys[valid]
    depth_values = depth_values[valid] / camera_config.DEPTH_SCALE
    print("DEPTH VALUES DEBUG")
    print("shape:", depth_values.shape)
    print("mean:", np.mean(depth_values))
    print("median:", np.median(depth_values))
    print("std:", np.std(depth_values))
    print("first_50:", depth_values[:50])

    p25 = np.percentile(depth_values, 25)
    p75 = np.percentile(depth_values, 75)

    depth_stats = {
        "min": float(depth_values.min()),
        "p25": float(p25),
        "median": float(np.median(depth_values)),
        "p75": float(p75),
        "max": float(depth_values.max()),
    }

    # print("-" * 40)
    # print(np.median(depth_values))
    # print("-" * 40)

    # cv2.imshow("DEPTH", slam_dict["depth"])
    # cv2.waitKey(0)
    # cv2.destroyWindow("DEPTH")

    local_xs = (xs - camera_config.CX) * depth_values / camera_config.FX
    local_ys = (ys - camera_config.CY) * depth_values / camera_config.FY
    local_zs = depth_values

    local_pos = np.array([
        np.median(local_xs),
        np.median(local_ys),
        np.median(local_zs)
    ])
    median_depth = local_pos[2]

    cx = int(np.median(xs))
    cy = int(np.median(ys))

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

    # world_pos = rotation.inv().apply(local_pos) + translation
    world_pos = rotation.apply(local_pos) + translation

    if any(v is None for v in world_pos):
        return None
    if any(np.isnan(v) for v in world_pos):
        return None

    return (
        world_pos,
        (cx, cy),
        local_pos,
        translation,
        median_depth,
        depth_stats,
    )
