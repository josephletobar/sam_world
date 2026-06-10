import os
import cv2
import numpy as np
import traceback
from dataclasses import dataclass
from utils.load_data import load_data

from modules.GraphBuilder import GraphBuilder
from modules.GraphChat import ChatWithGraph
from modules.SceneUnderstanding import SceneUnderstanding
from modules.SceneDiff import SceneDifferenceDetector
from modules.PriorityObjectDetector import PriorityObjectDetector
from modules.ObjectPerception import ObjectPerception
from modules.Association import Association
from modules.ObjectTracker import TrackObject

DISPLAY_W = 1920
DISPLAY_H = 1080

STEP_FRAMES = 5

@dataclass
class SLAMFrame:
    rgb: np.ndarray
    depth: np.ndarray
    pose: dict

class RobotWorldModel:

    def __init__(self, source):
        self.frame_idx = 0        

        self.global_objects = []

        self.DEFAULT_LABELS = ["person"]
        self.vocabulary = set(self.DEFAULT_LABELS)

        self.cur_slam_frame = SLAMFrame(
            rgb=None,
            depth=None,
            pose=None,
        )

        self.scene_diff_detector = SceneDifferenceDetector(self.cur_slam_frame)
        self.scene_understanding = SceneUnderstanding(client="openai", slam_frame=self.cur_slam_frame)
        self.priority_object_detector = PriorityObjectDetector(self.DEFAULT_LABELS, self.cur_slam_frame)
        self.object_perception = ObjectPerception(self.cur_slam_frame)
        self.graph_builder = GraphBuilder()
        self.association = Association(self.global_objects, self.graph_builder)

        self.tracker = None
        self.track_result = None

        self.yolo_boxes = None

        self.rgb_files, self.depth_files, self.rgb_to_pose, self.rgb_to_depth, self.pose_data = load_data(source)

    def get_next_frame(self):

        if self.frame_idx >= len(self.rgb_files):
            return False

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

        self.cur_slam_frame.rgb = rgb_frame
        self.cur_slam_frame.depth = depth_frame
        self.cur_slam_frame.pose = pose

        self.frame_idx += 1

        return True

    def show_video(self, frame):

        annotated = frame.copy()

        if self.track_result is not None:
            track_annotated = self.track_result.get("annotated")

            if track_annotated is not None:
                annotated = cv2.resize(track_annotated, (DISPLAY_W, DISPLAY_H))

        annotated = cv2.resize(annotated, (DISPLAY_W, DISPLAY_H))

        if self.yolo_boxes is not None:
            yolo_frame = self.cur_slam_frame.rgb
            if yolo_frame is None:
                yolo_frame = frame

            scale_x = DISPLAY_W / yolo_frame.shape[1]
            scale_y = DISPLAY_H / yolo_frame.shape[0]

            for box in self.yolo_boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow("Video", annotated)
        cv2.waitKey(1)

    def run(self):

        ret = self.get_next_frame()

        if not ret:
            return False

        frame = self.cur_slam_frame.rgb

        if self.frame_idx != 1 and self.frame_idx % STEP_FRAMES != 0:
            return True
        
        if self.tracker is not None and self.tracker.initialized:
            self.track_result = self.tracker.track()

            for tracked_object in self.track_result["objects"]:
                node_id = tracked_object.node_id

                for i, obj in enumerate(self.global_objects):
                    if obj.node_id == node_id:
                        self.global_objects[i] = tracked_object
                        break

        else:
            self.track_result = None

        priority_objects = self.priority_object_detector.detect()
        self.yolo_boxes = self.priority_object_detector.yolo_boxes
        priority_changed = len(priority_objects) > 0
        if priority_changed:
            print("PRIORITY OBJECTS DETECTED:", priority_objects)
            self.vocabulary.update(priority_objects)
        priority_changed = False  # NOTE: not usng for debugging needs work

        # Detect scene changes
        scene_diff_changed = self.scene_diff_detector.should_reprompt()
        scene_changed = priority_changed or scene_diff_changed

        # Run VLM and SAM if significant change detected or first frame
        if scene_changed:

            self.priority_object_detector.prev_yolo_labels = None # Reset 

            self.sam_labels = self.scene_understanding.get_labels(self.vocabulary)

            print(self.sam_labels)
            full_labels = (
                self.DEFAULT_LABELS +
                self.sam_labels
            )
            self.vocabulary.update(self.sam_labels)

            if len(full_labels) == 0:
                self.show_video(frame)
                return True
            
            objects, annotated = self.object_perception.get_objects(full_labels)
            for obj in objects:
                self.association.update(obj)

            if len(objects) > 0:
                self.tracker = TrackObject(self.cur_slam_frame)
                self.tracker.initialize(objects)

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
    # world = RobotWorldModel(r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg1_xyz (1)\rgbd_dataset_freiburg1_xyz")
    # world = RobotWorldModel(r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg2_pioneer_slam\rgbd_dataset_freiburg2_pioneer_slam")
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
