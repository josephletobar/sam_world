import os
import cv2
import numpy as np
import traceback

from utils.association import association
from modules.GraphBuilder import GraphBuilder
from modules.GraphChat import ChatWithGraph
from modules.SceneUnderstanding import SceneUnderstanding
from modules.SceneDiff import SceneDifferenceDetector
from modules.PriorityObjectDetector import PriorityObjectDetector
from modules.ObjectPerception import ObjectPerception

class RobotWorldModel:

    def __init__(self, source):
        self.load_slam_data(source)
        self.frame_idx = 0        

        self.scene_diff_detector = SceneDifferenceDetector()
        self.scene_understanding = SceneUnderstanding(client="openai")
        self.priority_object_detector = PriorityObjectDetector()
        self.object_perception = ObjectPerception()
        self.graph_builder = GraphBuilder()

        self.DEFAULT_LABELS = [
        ]
        self.vocabulary = set(self.DEFAULT_LABELS)

        # Global object storage
        self.global_objects = []

        self.yolo_boxes = None

    def load_slam_data(self, source):

        self.rgb_files = sorted(
            os.path.join(source, "rgb", f)
            for f in os.listdir(os.path.join(source, "rgb"))
        )

        self.depth_files = sorted(
            os.path.join(source, "depth", f)
            for f in os.listdir(os.path.join(source, "depth"))
        )

        if len(self.rgb_files) == 0:
            raise ValueError("No RGB images found")

        if len(self.depth_files) == 0:
            raise ValueError("No depth images found")

        gt_path = os.path.join(source, "groundtruth.txt")

        self.pose_data = []
        with open(gt_path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.strip().split()

                if len(parts) < 8:
                    continue

                self.pose_data.append({
                    "timestamp": float(parts[0]),
                    "tx": float(parts[1]),
                    "ty": float(parts[2]),
                    "tz": float(parts[3]),
                    "qx": float(parts[4]),
                    "qy": float(parts[5]),
                    "qz": float(parts[6]),
                    "qw": float(parts[7]),
                })

        if len(self.pose_data) == 0:
            raise ValueError("No poses found")

        # load timestamps
        rgb_ts_path = os.path.join(source, "rgb.txt")
        depth_ts_path = os.path.join(source, "depth.txt")

        self.rgb_timestamps = np.loadtxt(rgb_ts_path)
        self.depth_timestamps = np.loadtxt(depth_ts_path)

        self.pose_timestamps = np.array([
            pose["timestamp"]
            for pose in self.pose_data
        ])

        # precompute alignments
        self.rgb_to_pose = []
        self.rgb_to_depth = []

        for rgb_ts in self.rgb_timestamps:

            pose_idx = np.searchsorted(
                self.pose_timestamps,
                rgb_ts
            )

            if pose_idx > 0 and (
                pose_idx == len(self.pose_timestamps)
                or abs(self.pose_timestamps[pose_idx - 1] - rgb_ts)
                < abs(self.pose_timestamps[pose_idx] - rgb_ts)
            ):
                pose_idx -= 1

            depth_idx = np.searchsorted(
                self.depth_timestamps,
                rgb_ts
            )

            if depth_idx > 0 and (
                depth_idx == len(self.depth_timestamps)
                or abs(self.depth_timestamps[depth_idx - 1] - rgb_ts)
                < abs(self.depth_timestamps[depth_idx] - rgb_ts)
            ):
                depth_idx -= 1

            self.rgb_to_pose.append(pose_idx)
            self.rgb_to_depth.append(depth_idx)

        print("RGB:", len(self.rgb_files))
        print("Depth:", len(self.depth_files))
        print("Poses:", len(self.pose_data))
        print("RGB->Pose:", len(self.rgb_to_pose))
        print("RGB->Depth:", len(self.rgb_to_depth))

        print(
            "First RGB->Pose dt:",
            abs(
                self.pose_timestamps[self.rgb_to_pose[0]]
                - self.rgb_timestamps[0]
            )
        )

        print(
            "Mean RGB->Pose dt:",
            np.mean([
                abs(self.pose_timestamps[p] - t)
                for p, t in zip(self.rgb_to_pose, self.rgb_timestamps)
            ])
        )

    def get_next_frame(self):

        if self.frame_idx >= len(self.rgb_files):
            return False, None, None

        rgb_idx = self.frame_idx
        depth_idx = self.rgb_to_depth[rgb_idx]
        pose_idx = self.rgb_to_pose[rgb_idx]

        rgb_frame = cv2.imread(
            self.rgb_files[rgb_idx]
        )

        depth_frame = cv2.imread(
            self.depth_files[depth_idx],
            cv2.IMREAD_UNCHANGED
        )

        if rgb_frame is None:
            raise ValueError(
                f"Could not read RGB frame: {self.rgb_files[rgb_idx]}"
            )

        if depth_frame is None:
            raise ValueError(
                f"Could not read depth frame: {self.depth_files[depth_idx]}"
            )

        if depth_frame.shape[:2] != rgb_frame.shape[:2]:
            depth_frame = cv2.resize(
                depth_frame,
                (rgb_frame.shape[1], rgb_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        pose = self.pose_data[pose_idx]

        slam_dict = {
            "rgb": rgb_frame,
            "depth": depth_frame,
            "pose": pose,
        }

        self.frame_idx += 1

        return True, rgb_frame, slam_dict
    
    def show_video(self, frame):

        annotated = frame.copy()

        if self.yolo_boxes is not None:
            for box in self.yolo_boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

        cv2.imshow("Video", annotated)
        cv2.waitKey(1)

    def run(self):

        ret, frame, slam_dict = self.get_next_frame()
    
        if not ret:
            return False
        
        pose = slam_dict["pose"]

        priority_objects = self.priority_object_detector.detect(frame)
        if len(priority_objects) > 0:
            print("PRIORITY OBJECTS DETECTED:", priority_objects)
            self.vocabulary.update(priority_objects)
        self.yolo_boxes = self.priority_object_detector.yolo_boxes

        # Detect scene changes and get YOLO boxes
        scene_changed = self.scene_diff_detector.should_reprompt(frame, pose)

        # Run VLM and SAM if significant change detected or first frame
        if scene_changed:
            print("--- VLM RAN ---")

            self.sam_labels = self.scene_understanding.get_labels(frame, self.vocabulary)

            print(self.sam_labels)
            full_labels = (
                self.DEFAULT_LABELS +
                self.sam_labels
            )
            self.vocabulary.update(self.sam_labels)

            if len(full_labels) == 0:
                self.show_video(frame)
                return True
            
            objects, annotated = self.object_perception.get_objects(frame, full_labels, slam_dict)

            if len(self.global_objects) > 0:
            
                for object in objects:
                    different, object.node_id = association(new_object=object, world_objects=self.global_objects)

                    if different:
                        self.global_objects.append(object)
                        self.graph_builder.add_object(object, self.global_objects)

                    elif not different:
                        # Merge Nodes (skip for now)
                        pass

            else:
                for obj in objects:
                    obj.node_id = f"{obj.label}_{sum(1 for o in self.global_objects if o.label == obj.label)}"
                    self.global_objects.append(obj)
                    self.graph_builder.add_object(obj, self.global_objects)

            self.graph_builder.draw_2d_graph()

            for obj in objects:
                cx, cy = obj.image_pos
                world_x, world_y, world_z = obj.world_pos

                cv2.putText(
                    annotated,
                    f"{world_x:.2f}, {world_y:.2f}, {world_z:.2f}",
                    (int(cx), int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
            )

            # Display the annotated frame
            self.show_video(annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):

                self.graph_builder.save_graph()

                return False

            self.graph_builder.save_graph()

            return True
        
        self.show_video(frame)
        return True
        

if __name__ == "__main__":

    # world = RobotWorldModel("assets/challenge_video.mp4")
    # world = RobotWorldModel(
    #     r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg1_xyz (1)\rgbd_dataset_freiburg1_xyz"
    # )
    # world = RobotWorldModel(
    #     r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg2_pioneer_slam\rgbd_dataset_freiburg2_pioneer_slam"
    # )
    # world = RobotWorldModel(r"D:\forest_data")
    world = RobotWorldModel(r"D:\kab3_data")


    try:
        while True:
            running = world.run()
            if not running:
                break

    except KeyboardInterrupt:
        print("Saving graph...")

    except Exception as e:
        print(type(e).__name__)
        print(e)
        traceback.print_exc()

    finally:

        final_graph = world.graph_builder.draw_3d_graph()

        chat = ChatWithGraph(final_graph)
        while True:
            chat.run()
