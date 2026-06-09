from cProfile import label
import os
import json
import cv2
import numpy as np
import traceback
from networkx.readwrite import json_graph
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from ultralytics.models.sam import SAM3SemanticPredictor
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

from utils.get_obj_pos import get_pos
from utils.clip_embedding import embed_image, embed_text
from utils.association import association

from modules.GraphChat import ChatWithGraph
from modules.SceneUnderstanding import SceneUnderstanding
from modules.SceneDiff import SceneDifferenceDetector
from modules.PriorityObjectDetector import PriorityObjectDetector
from modules.ObjectPerception import ObjectPerception

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

class RobotWorldModel:

    def __init__(self, source):

        # VIDEO
        if source.endswith(".mp4"):
            self.source_type = "video"
            self.cap = cv2.VideoCapture(source)

        # SLAM STREAM DIR
        else:
            self.source_type = "slam stream"
            self.load_slam_data(source)
            self.frame_idx = 0

        # SAM3 Setup
        overrides = dict(
            conf=0.8,
            task="segment",
            mode="predict",
            model="models/sam3.pt",
            save=False,
        )

        self.scene_diff_detector = SceneDifferenceDetector()
        self.scene_understanding = SceneUnderstanding(client="openai")
        self.priority_object_detector = PriorityObjectDetector()
        self.object_perception = ObjectPerception(sam_predictor=SAM3SemanticPredictor(overrides=overrides))

        # Graph SETUP
        self.G = nx.Graph()
        self.pos = {}

        # Set Constants/Thresholds
        self.SAM_STEP = 5
        self.CHANGE_STEP = 5
        self.DIFF_TRESHOLD = 0.75

        self.DEFAULT_LABELS = [
        ]
        self.vocabulary = set(self.DEFAULT_LABELS)

        # Init Variables
        self.run_gpt = False
        self.sam_labels = []
        self.prev_gpt_call = {
            "frame" : None,
            "position" : None
        }

        # Global object storage
        self.global_objects = []

        self.frame_count = 0 # frame tracking

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

        rgb_count = len(self.rgb_files)
        depth_count = len(self.depth_files)
        pose_count = len(self.pose_data)

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

    # Adds to Graph and updates memories
    def add_object(self, obj):

        print("--NEW OBJECT--")

        world_x, world_y, world_z = obj.world_pos

        self.G.add_node(
            obj.node_id,
            world_x=float(world_x),
            world_y=float(world_y),
            world_z=float(world_z),
            txt_embedding=obj.txt_embedding.tolist(),
            img_embedding=obj.img_embedding,
            confidence=obj.confidence
        )

        self.pos[obj.node_id] = (
            world_x,
            world_z
        )

        for other_obj in self.global_objects:
            if other_obj.node_id == obj.node_id:
                continue

            dist = np.linalg.norm(
                np.array(obj.world_pos) -
                np.array(other_obj.world_pos)
            )

            self.G.add_edge(
                obj.node_id,
                other_obj.node_id,
                weight=round(dist, 2)
            )

    def draw_graph(self):

        threshold_graph = nx.Graph(
            (u, v, d)
            for u, v, d in self.G.edges(data=True)
            if d["weight"] < 0.5
        )

        mst = nx.minimum_spanning_tree(self.G, weight="weight")

        final_graph = nx.compose(mst, threshold_graph)

        edge_labels = nx.get_edge_attributes(final_graph, "weight")

        node_colors = [
            self.G.nodes[n]["cluster"]
            for n in final_graph.nodes()
        ]

        nx.draw(
            final_graph,
            self.pos,
            with_labels=True,
            node_size=1000,
            node_color=node_colors,
            cmap=plt.cm.tab10
        )

        # nx.draw(final_graph, self.pos, with_labels=True, node_size=1000)

        nx.draw_networkx_edge_labels(
            final_graph,
            self.pos,
            edge_labels=edge_labels
        )

        return final_graph
    
    
    def cluster(self):
        nodes = list(self.G.nodes())

        X = np.array([
            [
                self.G.nodes[n]["world_x"],
                self.G.nodes[n]["world_y"],
                self.G.nodes[n]["world_z"],
            ]
            for n in nodes
        ])

        if len(X) == 0:
            return

        labels = DBSCAN(
            eps=0.4,
            min_samples=3
        ).fit_predict(X)

        for node, cluster_id in zip(nodes, labels):
            self.G.nodes[node]["cluster"] = int(cluster_id)

    def get_next_frame(self):

        if self.source_type == "video":
            ret, frame = self.cap.read()
            return ret, frame, None

        elif self.source_type == "slam stream":

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

    def draw_yolo_boxes(self, frame, yolo_boxes):
        if yolo_boxes is None:
            return frame

        annotated = frame.copy()
        for box in yolo_boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

        return annotated


    def run(self):

        ret, frame, slam_dict = self.get_next_frame()
        pose = slam_dict["pose"]
        depth = slam_dict["depth"]

        if not ret:
            return False

        self.frame_count += 1

        priority_objects = self.priority_object_detector.detect(frame)
        if len(priority_objects) > 0:
            print("PRIORITY OBJECTS DETECTED:", priority_objects)
            self.vocabulary.update(priority_objects)
        yolo_boxes = self.priority_object_detector.yolo_boxes

        # # Only run SAM every SAM_STEP frames
        # if self.frame_count % self.SAM_STEP != 0:
        #     return True

        self.run_gpt = False

        yolo_boxes = None

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
                cv2.imshow("Video", self.draw_yolo_boxes(frame, yolo_boxes))
                cv2.waitKey(1)
                return True
            
            objects, annotated = self.object_perception.get_objects(frame, full_labels, slam_dict)

            if len(self.global_objects) > 0:
            
                for object in objects:
                    different, object.node_id = association(new_object=object, world_objects=self.global_objects)

                    if different:
                        self.global_objects.append(object)
                        self.add_object(object)

                    elif not different:
                        # Merge Nodes (skip for now)
                        pass

            else:
                for obj in objects:
                    obj.node_id = f"{obj.label}_{sum(1 for o in self.global_objects if o.label == obj.label)}"
                    self.global_objects.append(obj)
                    self.add_object(obj)

            self.cluster()

            # Display graph using NetworkX and Matplotlib
            plt.clf()

            self.draw_graph()

            plt.pause(0.1)
            plt.draw()

            for obj in objects:
                cx, cy = obj.image_pos
                world_x, world_y, world_z = obj.world_pos

                cv2.circle(
                    annotated,
                    (int(cx), int(cy)),
                    5,
                    (0, 0, 0),
                    -1
                )

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
            annotated = self.draw_yolo_boxes(annotated, yolo_boxes)
            cv2.imshow("Video", annotated)
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("q"):

                final_graph = self.draw_graph()

                data = json_graph.node_link_data(final_graph)
                with open("graph.json", "w") as f:
                    json.dump(data, f, indent=2)

                return False

            final_graph = self.draw_graph()

            data = json_graph.node_link_data(final_graph)
            with open("graph.json", "w") as f:
                json.dump(data, f, indent=2)

            return True
        
        cv2.imshow("Video", self.draw_yolo_boxes(frame, yolo_boxes))
        cv2.waitKey(1)
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

        final_graph = world.draw_graph()

        plt.close("all")


        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for node in final_graph.nodes():
            x = final_graph.nodes[node]["world_x"]
            y = final_graph.nodes[node]["world_y"]
            z = final_graph.nodes[node]["world_z"]

            cluster = final_graph.nodes[node].get("cluster", -1)
            
            color = plt.cm.tab10(cluster % 10)
            ax.scatter(
                x, y, z,
                color=color,
                s=150
            )

            print(node, cluster)

            ax.text(x, y, z, node)

        for u, v in final_graph.edges():
            x1 = final_graph.nodes[u]["world_x"]
            y1 = final_graph.nodes[u]["world_y"]
            z1 = final_graph.nodes[u]["world_z"]

            x2 = final_graph.nodes[v]["world_x"]
            y2 = final_graph.nodes[v]["world_y"]
            z2 = final_graph.nodes[v]["world_z"]

            ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", alpha=0.4)

            weight = final_graph[u][v]["weight"]

            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                (z1 + z2) / 2,
                f"{weight:.2f}"
            )

        data = json_graph.node_link_data(final_graph)
        with open("graph.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Graph saved.")

        plt.show(block=False)

        chat = ChatWithGraph(final_graph)
        while True:
            chat.run()
