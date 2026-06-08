import os
import json
from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
from openai import OpenAI
import base64
import random
import numpy as np
import traceback
from scripts.vlm import vlm
from scripts.get_obj_pos import get_pos
from scripts.scene_dif import should_reprompt
from scripts.graph_chat import ChatWithGraph
from scripts.clip_embedding import embed_image, embed_text
from scripts.association import association
from networkx.readwrite import json_graph
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLOWorld


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")



class SamWorld:

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
            model="sam3.pt",
        )
        self.sam_predictor = SAM3SemanticPredictor(overrides=overrides)

        self.yolo_model = YOLOWorld("yolov8x-worldv2.pt")

        # Graph SETUP
        self.G = nx.Graph()
        self.pos = {}

        # Set Constants/Thresholds
        self.SAM_STEP = 5
        self.CHANGE_STEP = 5
        self.SIM_THRESHOLD = 0.8
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

        # In memory storage (index alligned)
        self.labels = []
        self.embedding_matrix = [] # img embeddings
        self.segmented_rgbs = [] # rgb images
        self.world_poses = [] 

        self.segmented_depths = [] # depth channel
        self.poses = [] # depth channel

        self.label_counts = {}

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
    def add_object(
        self,
        txt_embedding,
        img_embedding,
        node_id,
        img,
        world_pos,
        confidence,
    ):

        print("--NEW OBJECT--")

        self.embedding_matrix.append(img_embedding)
        self.labels.append(node_id)
        self.segmented_rgbs.append(img)

        self.world_poses.append(world_pos)

        world_x, world_y, world_z = world_pos


        self.G.add_node(
            node_id,
            world_x=float(world_x),
            world_y=float(world_y),
            world_z=float(world_z),
            txt_embedding=txt_embedding.tolist(),
            img_embedding=img_embedding,
            confidence=confidence
        )

        self.pos[node_id] = (
            world_x,
            world_z
        )

        for other_id, other_pos in zip(
            self.labels,
            self.world_poses
        ):
            if node_id != other_id: 
                dist = np.linalg.norm(
                    np.array(world_pos) -
                    np.array(other_pos)
                )

                self.G.add_edge(
                    node_id,
                    other_id,
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


    def run(self):

        ret, frame, slam_dict = self.get_next_frame()
        pose = slam_dict["pose"]
        depth = slam_dict["depth"]

        if not ret:
            return False

        self.frame_count += 1

        # yolo_results = self.yolo_model.predict(
        #     frame,
        #     conf=0.7,
        #     verbose=False
        # )

        # Only run SAM every SAM_STEP frames
        if self.frame_count % self.SAM_STEP != 0:
            return True

        self.run_gpt = False

        # Stepped Frame Differencing Logic

        # first iteration
        if (
            self.prev_gpt_call["frame"] is None
            or
            self.prev_gpt_call["position"] is None
        ):

            self.prev_gpt_call["frame"] = frame
            self.prev_gpt_call["position"] = pose

            self.run_gpt = True

        elif self.frame_count % self.CHANGE_STEP == 0:

            prev_cur_frames =  self.prev_gpt_call["frame"], frame 
            prev_cur_pos = self.prev_gpt_call["position"], pose

            self.run_gpt, _, _ = should_reprompt(prev_cur_frames, prev_cur_pos)

            if self.run_gpt == True:
                self.prev_gpt_call["frame"] = frame
                self.prev_gpt_call["position"] = pose

        # Run LLM if significant change detected or first frame
        if self.run_gpt:
            print("--- LLM RUNNING ---")
            self.sam_labels = vlm(
                frame,
                list(self.vocabulary)
            )
            print(self.sam_labels)


        full_labels = (
            self.DEFAULT_LABELS +
            self.sam_labels
        )


        self.vocabulary.update(self.sam_labels)

        # Run SAM3 with the combined labels
        results = self.sam_predictor(
            frame,
            text=full_labels,
            # imgsz=448,
            save=False,
            verbose=False
        )
        result = results[0]
        annotated = result.plot()

        # Add Detected Labels to Knowledge Graph
        object_poses_buf = []
        for i, box in enumerate(result.boxes):
            # Results object
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            confidence = float(box.conf[0])
            mask = result.masks.data[i]
            mask = mask.cpu().numpy()

            # Find objects position
            ys, xs = np.where(mask > 0.5)

            object_pose = get_pos(mask, depth, pose)
            if object_pose is None: continue

            world_pos, image_pos = object_pose

            object_poses_buf.append((image_pos, world_pos, confidence))

            segmented_rgb = frame.copy()
            segmented_rgb[mask == 0] = 0
            segmented_rgb = segmented_rgb[
                ys.min():ys.max(),
                xs.min():xs.max()
            ]

            img_embedding = embed_image(segmented_rgb)
            txt_embedding = embed_text(label)

            # Object Association
            new_data = (label, img_embedding, segmented_rgb, world_pos if slam_dict else None)
            all_data = (self.labels, self.embedding_matrix, self.segmented_rgbs, self.world_poses if slam_dict else None)

            if len(self.embedding_matrix) > 0:

                best_prob, best_id, best_idx = association(new_data, all_data)
        
                # New Node
                # print("-------")
                # print(best_prob)
                # print("-------")
                if best_prob < self.SIM_THRESHOLD:

                    count = self.label_counts.get(label, 0)
                    node_id = f"{label}_{count}"
                    self.label_counts[label] = count + 1

                    self.add_object(
                        txt_embedding,
                        img_embedding,
                        node_id,
                        segmented_rgb,
                        world_pos,
                        confidence
                    )

                    # cv2.imshow("NEWOBJECT", segmented_rgb)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow("NEWOBJECT")
                
                else: # Existing Node
                    # Merge Nodes

                    # # debug to visualize the matched objects
                    # print(best_prob)
                    # cv2.imshow("match1", segmented)
                    # cv2.imshow("match2", self.segmented_rgbs[best_idx])
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    pass

            # First Node
            else:
                count = self.label_counts.get(label, 0)
                node_id = f"{label}_{count}"
                self.label_counts[label] = count + 1

                self.add_object(
                    txt_embedding,
                    img_embedding,
                    node_id,
                    segmented_rgb,
                    world_pos,
                    confidence
                )

        self.cluster()

        # Display graph using NetworkX and Matplotlib
        plt.clf()

        # mst = nx.minimum_spanning_tree(self.G, weight="weight")
        # edge_labels = nx.get_edge_attributes(mst, "weight")

        # nx.draw(mst, self.pos, with_labels=True, node_size=1000)

        # nx.draw_networkx_edge_labels(
        #     mst,
        #     self.pos,
        #     edge_labels=edge_labels
        # )

        self.draw_graph()

        plt.pause(0.1)
        plt.draw()

        for ((cx, cy), world_pos, confidence) in object_poses_buf:
            world_x, world_y, world_z = world_pos

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
        cv2.imshow("SAM3 Video", annotated)

        # cv2.waitKey(0)
        # cv2.destroyWindow("SAM3 Video")

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
    


if __name__ == "__main__":

    # world = SamWorld("assets/challenge_video.mp4")
    # world = SamWorld(
    #     r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg1_xyz (1)\rgbd_dataset_freiburg1_xyz"
    # )
    # world = SamWorld(
    #     r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg2_pioneer_slam\rgbd_dataset_freiburg2_pioneer_slam"
    # )
    # world = SamWorld(r"D:\forest_data")
    world = SamWorld(r"D:\kab3_data")


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