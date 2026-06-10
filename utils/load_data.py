import os
import numpy as np

def load_data(source):

        rgb_files = sorted(
            os.path.join(source, "rgb", f)
            for f in os.listdir(os.path.join(source, "rgb"))
        )

        depth_files = sorted(
            os.path.join(source, "depth", f)
            for f in os.listdir(os.path.join(source, "depth"))
        )

        if len(rgb_files) == 0:
            raise ValueError("No RGB images found")

        if len(depth_files) == 0:
            raise ValueError("No depth images found")

        gt_path = os.path.join(source, "groundtruth.txt")

        pose_data = []
        with open(gt_path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.strip().split()

                if len(parts) < 8:
                    continue

                pose_data.append({
                    "timestamp": float(parts[0]),
                    "tx": float(parts[1]),
                    "ty": float(parts[2]),
                    "tz": float(parts[3]),
                    "qx": float(parts[4]),
                    "qy": float(parts[5]),
                    "qz": float(parts[6]),
                    "qw": float(parts[7]),
                })

        if len(pose_data) == 0:
            raise ValueError("No poses found")

        # load timestamps
        rgb_ts_path = os.path.join(source, "rgb.txt")
        depth_ts_path = os.path.join(source, "depth.txt")

        rgb_timestamps = np.loadtxt(rgb_ts_path)
        depth_timestamps = np.loadtxt(depth_ts_path)

        pose_timestamps = np.array([
            pose["timestamp"]
            for pose in pose_data
        ])

        # precompute alignments
        rgb_to_pose = []
        rgb_to_depth = []

        for rgb_ts in rgb_timestamps:

            pose_idx = np.searchsorted(
                pose_timestamps,
                rgb_ts
            )

            if pose_idx > 0 and (
                pose_idx == len(pose_timestamps)
                or abs(pose_timestamps[pose_idx - 1] - rgb_ts)
                < abs(pose_timestamps[pose_idx] - rgb_ts)
            ):
                pose_idx -= 1

            depth_idx = np.searchsorted(
                depth_timestamps,
                rgb_ts
            )

            if depth_idx > 0 and (
                depth_idx == len(depth_timestamps)
                or abs(depth_timestamps[depth_idx - 1] - rgb_ts)
                < abs(depth_timestamps[depth_idx] - rgb_ts)
            ):
                depth_idx -= 1

            rgb_to_pose.append(pose_idx)
            rgb_to_depth.append(depth_idx)

        print("RGB:", len(rgb_files))
        print("Depth:", len(depth_files))
        print("Poses:", len(pose_data))
        print("RGB->Pose:", len(rgb_to_pose))
        print("RGB->Depth:", len(rgb_to_depth))

        print(
            "First RGB->Pose dt:",
            abs(
                pose_timestamps[rgb_to_pose[0]]
                - rgb_timestamps[0]
            )
        )

        print(
            "Mean RGB->Pose dt:",
            np.mean([
                abs(pose_timestamps[p] - t)
                for p, t in zip(rgb_to_pose, rgb_timestamps)
            ])
        )

        return rgb_files, depth_files, rgb_to_pose, rgb_to_depth, pose_data