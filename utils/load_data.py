import os
import numpy as np
import re
from utils.camera_config import configure_camera_intrinsics

ZED_DEPTH_SCALE = 1000.0


def _configure_camera_from_source(source):
        intrinsics_path = os.path.join(source, "camera_intrinsics.txt")

        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Missing camera intrinsics file: {intrinsics_path}")

        with open(intrinsics_path, "r") as f:
            text = f.read()

        match = re.search(
            r"zed2i_depth_caminfo .*?fx=([^ ]+) fy=([^ ]+) cx=([^ ]+) cy=([^ ]+)",
            text
        )
        if match is None:
            match = re.search(
                r"zed2i_right_caminfo .*?fx=([^ ]+) fy=([^ ]+) cx=([^ ]+) cy=([^ ]+)",
                text
            )

        if match is None:
            raise ValueError(f"No usable intrinsics found in {intrinsics_path}")

        fx, fy, cx, cy = [float(value) for value in match.groups()]
        configure_camera_intrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=ZED_DEPTH_SCALE
        )
        print(
            "Camera intrinsics:",
            f"fx={fx}",
            f"fy={fy}",
            f"cx={cx}",
            f"cy={cy}"
        )
        print(f"Depth scale: {ZED_DEPTH_SCALE}")


def _load_timestamped_files(source, filename, fallback_dir):
        path = os.path.join(source, filename)
        timestamps = []
        files = []
        has_listed_files = False

        with open(path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < 1:
                    continue

                timestamps.append(float(parts[0]))

                if len(parts) >= 2:
                    has_listed_files = True
                    files.append(os.path.join(source, parts[1]))
                else:
                    files.append(None)

        if len(timestamps) == 0:
            raise ValueError(f"No timestamps found in {path}")

        if not has_listed_files:
            folder = os.path.join(source, fallback_dir)
            files = [
                os.path.join(folder, name)
                for name in sorted(os.listdir(folder))
                if not name.startswith("._")
            ]

            if len(files) < len(timestamps):
                raise ValueError(
                    f"{folder} has {len(files)} files but {path} has "
                    f"{len(timestamps)} timestamps"
                )

            files = files[:len(timestamps)]

        return files, np.array(timestamps)


def load_data(source):
        source = os.path.abspath(source)
        print(f"Loading dataset: {source}")
        _configure_camera_from_source(source)

        rgb_files, rgb_timestamps = _load_timestamped_files(
            source,
            "rgb.txt",
            "rgb"
        )

        depth_files, depth_timestamps = _load_timestamped_files(
            source,
            "depth.txt",
            "depth"
        )

        if len(rgb_files) == 0:
            raise ValueError("No RGB images found")

        if len(depth_files) == 0:
            raise ValueError("No depth images found")

        missing_rgb = [
            path
            for path in rgb_files[:10] + rgb_files[-10:]
            if not os.path.exists(path)
        ]
        if missing_rgb:
            raise FileNotFoundError(f"Missing RGB files, first missing: {missing_rgb[0]}")

        missing_depth = [
            path
            for path in depth_files[:10] + depth_files[-10:]
            if not os.path.exists(path)
        ]
        if missing_depth:
            raise FileNotFoundError(f"Missing depth files, first missing: {missing_depth[0]}")

        if not np.all(np.diff(rgb_timestamps) > 0):
            raise ValueError("RGB timestamps must be strictly increasing")

        if not np.all(np.diff(depth_timestamps) > 0):
            raise ValueError("Depth timestamps must be strictly increasing")

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

        pose_timestamps = np.array([
            pose["timestamp"]
            for pose in pose_data
        ])

        if not np.all(np.diff(pose_timestamps) > 0):
            raise ValueError("Pose timestamps must be strictly increasing")

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
