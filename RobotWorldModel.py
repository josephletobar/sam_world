import os
import cv2
import numpy as np
import traceback
from dataclasses import dataclass
from datetime import datetime
from utils.load_data import load_data
from utils.video_recorder import VideoRecorder

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
SIDE_BY_SIDE_W = 1920
SIDE_BY_SIDE_H = 1080

STEP_FRAMES = 10

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
        self.scene_vocab = set()

        self.cur_slam_frame = SLAMFrame(
            rgb=None,
            depth=None,
            pose=None,
        )

        self.cur_scene = None

        self.timestamp = int(datetime.now().timestamp() * 1000)
        self.output_dir = f"examples/{self.timestamp}"

        self.scene_diff_detector = SceneDifferenceDetector(self.cur_slam_frame, threshold=0.55)
        self.scene_understanding = SceneUnderstanding(client="openai", slam_frame=self.cur_slam_frame, vocab=self.vocabulary, scene_vocab=self.scene_vocab)
        self.priority_object_detector = PriorityObjectDetector(self.DEFAULT_LABELS, self.cur_slam_frame)
        self.object_perception = ObjectPerception(self.cur_slam_frame)
        self.scene_recorder = VideoRecorder(f"{self.output_dir}/scene.mp4", fps=20)
        self.graph_recorder = VideoRecorder(f"{self.output_dir}/graph_2d.mp4", fps=20)
        self.side_by_side_recorder = VideoRecorder(f"{self.output_dir}/side_by_side.mp4", fps=20)
        self.graph_builder = GraphBuilder(
            recorder=self.graph_recorder,
            graph_path=f"{self.output_dir}/graph.json"
        )
        self.association = Association(self.global_objects, self.graph_builder, threshold=0.575)

        self.tracker = None
        self.track_result = None

        self.yolo_boxes = None
        self.closed = False

        self.rgb_files, self.depth_files, self.rgb_to_pose, self.rgb_to_depth, self.pose_data = load_data(source)

    def _fit_frame(self, frame, width, height):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        frame_h, frame_w = frame.shape[:2]
        scale = min(width / frame_w, height / frame_h)
        resized_w = max(1, int(round(frame_w * scale)))
        resized_h = max(1, int(round(frame_h * scale)))
        resized = cv2.resize(frame, (resized_w, resized_h))
        x = (width - resized_w) // 2
        y = (height - resized_h) // 2
        canvas[y:y + resized_h, x:x + resized_w] = resized
        return canvas

    def _write_side_by_side(self, scene_frame):
        graph_frame = self.graph_builder.get_2d_graph_frame()
        panel_w = SIDE_BY_SIDE_W // 2

        scene_panel = self._fit_frame(scene_frame, panel_w, SIDE_BY_SIDE_H)
        graph_panel = self._fit_frame(graph_frame, SIDE_BY_SIDE_W - panel_w, SIDE_BY_SIDE_H)
        combined = np.hstack((scene_panel, graph_panel))

        self.side_by_side_recorder.write(combined)

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

        if depth_frame.ndim > 2:
            depth_frame = depth_frame[:, :, 0]

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

        if self.cur_scene:
            scene_text = str(self.cur_scene)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.1
            thickness = 2
            padding_x = 14
            padding_y = 10

            (text_w, text_h), baseline = cv2.getTextSize(
                scene_text,
                font,
                font_scale,
                thickness
            )

            text_x = (annotated.shape[1] - text_w) // 2
            text_y = 42
            rect_x1 = max(text_x - padding_x, 0)
            rect_y1 = max(text_y - text_h - padding_y, 0)
            rect_x2 = min(text_x + text_w + padding_x, annotated.shape[1] - 1)
            rect_y2 = min(text_y + baseline + padding_y, annotated.shape[0] - 1)

            cv2.rectangle(
                annotated,
                (rect_x1, rect_y1),
                (rect_x2, rect_y2),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                annotated,
                scene_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )

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
        self.scene_recorder.write(annotated)
        self.graph_builder.write_2d_graph_frame()
        self._write_side_by_side(annotated)
        cv2.waitKey(1)

    def close(self):
        if self.closed:
            return

        self.scene_recorder.release()
        self.graph_recorder.release()
        self.side_by_side_recorder.release()
        cv2.destroyAllWindows()
        self.closed = True

    def run(self):

        ret = self.get_next_frame()

        if not ret:
            return False
        
        print(self.vocabulary)

        frame = self.cur_slam_frame.rgb

        if self.frame_idx != 1 and self.frame_idx % STEP_FRAMES != 0:
            self.show_video(frame)
            return True
        
        if self.tracker is not None and self.tracker.initialized:
            self.track_result = self.tracker.track()

            tracked_objects = self.track_result["objects"]
            if len(tracked_objects) > 0:
                for tracked_object in tracked_objects:
                    node_id = tracked_object.node_id

                    for i, obj in enumerate(self.global_objects):
                        if obj.node_id == node_id:
                            tracked_object.first_seen = obj.first_seen
                            tracked_object.last_seen = self.cur_slam_frame.pose.get("timestamp")
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

        # Detect scene changes
        scene_diff_changed = self.scene_diff_detector.should_reprompt()
        scene_changed = priority_changed or scene_diff_changed

        # Run VLM and SAM if significant change detected or first frame
        if scene_changed:

            self.priority_object_detector.prev_yolo_labels = None # Reset 

            self.sam_labels, self.cur_scene = self.scene_understanding.get_labels()
            if self.cur_scene:
                self.scene_vocab.add(self.cur_scene)

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
                self.association.update(
                    obj,
                    timestamp=self.cur_slam_frame.pose.get("timestamp")
                )

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
    world = None
    # world = RobotWorldModel(r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg1_xyz (1)\rgbd_dataset_freiburg1_xyz")
    # world = RobotWorldModel(r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg2_pioneer_slam\rgbd_dataset_freiburg2_pioneer_slam")
    # world = RobotWorldModel(r"D:\forest_data")
    # world = RobotWorldModel(r"D:\kab3_data")
    world= RobotWorldModel(r"D:\dataset")

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
        if world is None:
            raise SystemExit

        world.close()
        world.graph_builder.save_graph()
        final_graph = world.graph_builder.draw_3d_graph()

        chat = ChatWithGraph(final_graph)
        while True:
            chat.run()
