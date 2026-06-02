import os
import json
from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
from openai import OpenAI
import base64
import random
import numpy as np
from dotenv import find_dotenv, load_dotenv
from scripts.llm import llm
from scripts.clip_embedding import embed
from scripts.association import association
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

load_dotenv(override=True)

class SamWorld:

    def __init__(self, source):

        # VIDEO
        if source.endswith(".mp4"):
            self.source_type = "video"
            self.cap = cv2.VideoCapture(source)

        # SLAM STREAM DIR
        else:
            self.source_type = "slam stream"
            self.rgb_files = sorted([
                os.path.join(source, "rgb", f)
                for f in os.listdir(os.path.join(source, "rgb"))
            ])

            self.depth_files = sorted([
                os.path.join(source, "depth", f)
                for f in os.listdir(os.path.join(source, "depth"))
            ])

            gt_path = os.path.join(source, "groundtruth.txt")
            with open(gt_path, "r") as f:
                self.pose_data = []
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
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

            self.frame_idx = 0

        # LLM Setup
        self.client = OpenAI()

        # SAM3 Setup
        overrides = dict(
            conf=0.8,
            task="segment",
            mode="predict",
            model="sam3.pt",
        )
        self.sam_predictor = SAM3SemanticPredictor(overrides=overrides)

        # Graph Display Setup
        plt.ion()
        plt.figure(figsize=(6, 6))
        plt.show(block=False)
        self.G = nx.Graph()
        self.pos = {}

        # JSON-Based Graph Setup
        self.GRAPH_PATH = "graph.json"
        self.graph = {
            "nodes": [],
            "edges": []
        }
        with open(self.GRAPH_PATH, "w") as f:
            json.dump(self.graph, f, indent=2)

        self.frame_count = 0 # frame tracking


        # Set Constants/Thresholds
        self.SAM_STEP = 5
        self.CHANGE_STEP = 5
        self.SIM_THRESHOLD = 0.9
        self.DIFF_TRESHOLD = 0.8

        self.DEFAULT_LABELS = [
            "road",
            "car",
            "tree"
        ]

        # Init Variables
        self.run_gpt = False
        self.prev_frame = None
        self.sam_labels = []

        # In memory storage (index alligned)
        self.node_ids = []
        self.embedding_matrix = [] # img embeddings
        self.segmented_rgbs = [] # rgb images
        self.segmented_depths = [] # depth channel
        self.poses = [] # depth channel

    # Adds to Graph and updates memories
    def add_node(
        self,
        node,
        embedding,
        node_id,
        img,
        depth = None,
        pose = None
    ):

        self.graph["nodes"].append(node)

        self.embedding_matrix.append(embedding)
        self.node_ids.append(node_id)
        self.segmented_rgbs.append(img)

        if depth is not None:
            self.segmented_depths.append(depth)
        if pose is not None:
            self.poses.append(pose)

        self.G.add_node(node_id)

        if len(self.G.nodes) > 1:

            other_nodes = [
                n for n in self.G.nodes
                if n != node_id
            ]

            self.G.add_edge(
                node_id,
                random.choice(other_nodes)
            )

        print("NEW OBJECT !!!!!!!!!!!!!!!!!!!!!!!")

    def frame_dif(self, frame):

        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_curr, gray_prev)
        score = diff.mean()

        # print(f"DIFFERENCE SCORE: {score}")

        if score > self.DIFF_TRESHOLD:
            run_gpt = True
        else:
            run_gpt = False

        return run_gpt

    def get_next_frame(self):

        # VIDEO
        if self.source_type == "video":
            ret, frame = self.cap.read()
            return ret, frame, None

        # IMAGE SEQUENCE
        elif self.source_type == "slam stream":
            if self.frame_idx >= len(self.rgb_files):
                return False, None

            rgb_frame = cv2.imread(
                self.rgb_files[self.frame_idx]
            )
            depth_frame = cv2.imread(
                self.depth_files[self.frame_idx]
            )
            depth_frame = cv2.resize(
                depth_frame,
                (640, 360)
            )
            pose = self.pose_data[self.frame_idx]

            slam_dict = {
                "rgb" : rgb_frame,
                "depth" : depth_frame,
                "pose" : pose,
            }

            self.frame_idx += 1

            return True, rgb_frame, slam_dict


    def run(self):

        ret, frame, slam_dict = self.get_next_frame()

        if not ret:
            return False

        self.frame_count += 1

        # Only run SAM every SAM_STEP frames
        if self.frame_count % self.SAM_STEP != 0:
            return True

        # Stepped Frame Differencing Logic
        if self.prev_frame is None:
            self.prev_frame = frame
            self.run_gpt = True
            score = None

        elif self.frame_count % self.CHANGE_STEP == 0:
            self.run_gpt = self.frame_dif(frame)

        # Prepare image for LLM
        frame = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode(".jpg", frame)
        base64_image = base64.b64encode(
            buffer
        ).decode("utf-8")

        # Run LLM if significant change detected or first frame
        if self.run_gpt:
            print("--- LLM RUNNING ---")
            self.sam_labels = llm(
                self.client,
                base64_image,
                self.DEFAULT_LABELS
            )
            print(self.sam_labels)
        full_labels = (
            self.DEFAULT_LABELS +
            self.sam_labels
        )

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

        # Add detected labels to graph
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            node_id = (
                f"{label}_{len(self.graph['nodes'])}"
            )

            mask = result.masks.data[i]
            mask = mask.cpu().numpy()

            ys, xs = np.where(mask > 0.5)

            segmented_depth = slam_dict["depth"].copy()
            segmented_depth[mask == 0] = 0
            segmented_depth = segmented_depth[
                ys.min():ys.max(),
                xs.min():xs.max()
            ]

            segmented_rgb = frame.copy()
            segmented_rgb[mask == 0] = 0
            segmented_rgb = segmented_rgb[
                ys.min():ys.max(),
                xs.min():xs.max()
            ]

            embedding = embed(segmented_rgb)


            # JSON-Based Graph Update
            node = {
                "id": node_id,
                "label": label,
                "embedding": embedding,
            }

            # Node/Object Association
            new_data = (node_id, embedding, segmented_rgb, segmented_depth if slam_dict else None, slam_dict["pose"] if slam_dict else None)
            all_data = (self.node_ids, self.embedding_matrix, self.segmented_rgbs, self.segmented_depths if slam_dict else None, self.poses if slam_dict else None)

            if len(self.embedding_matrix) > 0:

                best_prob, best_id, best_idx = association(new_data, all_data)
        
                # New Node
                if best_prob < self.SIM_THRESHOLD:
                    self.add_node(
                        node,
                        embedding,
                        node_id,
                        segmented_rgb,
                        segmented_depth if slam_dict else None,
                        slam_dict["pose"] if slam_dict else None
                    )
                # Existing Node
                else:
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
                self.add_node(
                    node,
                    embedding,
                    node_id,
                    segmented_rgb,
                    segmented_depth if slam_dict else None,
                    slam_dict["pose"] if slam_dict else None
                )

            with open(self.GRAPH_PATH, "w") as f:
                json.dump(
                    self.graph,
                    f,
                    indent=2
                )

        # Display graph using NetworkX and Matplotlib
        plt.clf()

        if len(self.pos) == 0:

            self.pos = nx.spring_layout(self.G)

        else:

            self.pos = nx.spring_layout(
                self.G,
                pos=self.pos
            )

        nx.draw(
            self.G,
            self.pos,
            with_labels=True,
            node_size=500,
            font_size=8
        )

        plt.pause(0.1)
        plt.draw()

        # Display the annotated frame
        cv2.imshow("SAM3 Video", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        return True


if __name__ == "__main__":

    # world = SamWorld("assets/challenge_video.mp4")
    world = SamWorld(
        r"C:\Users\jletobar3\Downloads\rgbd_dataset_freiburg2_pioneer_slam\rgbd_dataset_freiburg2_pioneer_slam"
    )

    while True:

        running = world.run()

        if not running:
            break
