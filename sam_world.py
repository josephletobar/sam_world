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
from networkx.drawing.nx_pydot import graphviz_layout
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
        # plt.ion()
        # plt.figure(figsize=(6, 6))
        # plt.show(block=False)
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
        self.SIM_THRESHOLD = 0.8
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
        self.object_poses = [] 

        self.segmented_depths = [] # depth channel
        self.poses = [] # depth channel

    # Adds to Graph and updates memories
    def add_node(
        self,
        node,
        embedding,
        node_id,
        img,
        object_pos = None,
    ):

        self.graph["nodes"].append(node)

        self.embedding_matrix.append(embedding)
        self.node_ids.append(node_id)
        self.segmented_rgbs.append(img)

        if object_pos is not None:
            self.object_poses.append(object_pos)


        self.G.add_node(node_id)

        print("NEW OBJECT !!!!!!!!!!!!!!!!!!!!!!!")

        for other_id, other_pos in zip(
            self.node_ids,
            self.object_poses
        ):
            if node_id != other_id: 
                dist = np.linalg.norm(
                    np.array(object_pos) -
                    np.array(other_pos)
                )

                self.G.add_edge(
                    node_id,
                    other_id,
                    weight=int(dist)
                )

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

        return False

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
        object_poses_buf = []
        for i, box in enumerate(result.boxes):
            # Results object
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            node_id = (
                f"{label}_{len(self.graph['nodes'])}"
            )

            mask = result.masks.data[i]
            mask = mask.cpu().numpy()

            # Find objects position
            ys, xs = np.where(mask > 0.5)

            segmented_depth = slam_dict["depth"].copy()
            segmented_depth[mask == 0] = 0

            ys, xs = np.where(mask > 0.5)
            cx = xs.mean()
            cy = ys.mean()
            dists = (xs - cx)**2 + (ys - cy)**2
            best_idx = np.argmin(dists)
            cx = xs[best_idx]
            cy = ys[best_idx]

            depth_values = segmented_depth[:, :, 0]

            depth_value = depth_values[
                depth_values > 0
            ].mean()

            # cv2.imshow("DEPTH", slam_dict["depth"])
            # cv2.waitKey(0)
            # cv2.destroyWindow("DEPTH")

            # distance from center
            img_h, img_w = frame.shape[:2]
            dx = (cx - img_w / 2) / (img_w / 2)
            dy = (img_h / 2 - cy) / (img_h / 2)
            local_x = dx * depth_value
            local_y = dy * depth_value
            local_z = depth_value

            world_x = slam_dict["pose"]["tx"] + local_x
            world_y = slam_dict["pose"]["ty"] + local_y
            world_z = slam_dict["pose"]["tz"] + local_z

            object_pos = (
                world_x,
                world_y,
                world_z
            )

            object_poses_buf.append(((cx, cy), object_pos))

            # dont tight crop
            # segmented_depth = segmented_depth[
            #     ys.min():ys.max(),
            #     xs.min():xs.max()
            # ]

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
            new_data = (node_id, embedding, segmented_rgb, object_pos if slam_dict else None)
            all_data = (self.node_ids, self.embedding_matrix, self.segmented_rgbs, self.object_poses if slam_dict else None)

            if len(self.embedding_matrix) > 0:

                best_prob, best_id, best_idx = association(new_data, all_data)
        
                # New Node
                if best_prob < self.SIM_THRESHOLD:
                    self.add_node(
                        node,
                        embedding,
                        node_id,
                        segmented_rgb,
                        object_pos if slam_dict else None
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
                self.add_node(
                    node,
                    embedding,
                    node_id,
                    segmented_rgb,
                    object_pos if slam_dict else None
                )

            with open(self.GRAPH_PATH, "w") as f:
                json.dump(
                    self.graph,
                    f,
                    indent=2
                )

        # # Display graph using NetworkX and Matplotlib
        # plt.clf()

        # if len(self.pos) == 0:

        #     self.pos = nx.spring_layout(self.G)

        # else:

        #     self.pos = nx.spring_layout(
        #         self.G,
        #         pos=self.pos
        #     )

        # mst = nx.minimum_spanning_tree(self.G, weight="weight")

        # nx.draw(
        #     mst,
        #     self.pos,
        #     with_labels=True,
        #     node_size=500,
        #     font_size=8
        # )

        # plt.pause(0.1)
        # plt.draw()

        for ((cx, cy), object_pos) in object_poses_buf:

            world_x, world_y, world_z = object_pos
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

            mst = nx.minimum_spanning_tree(self.G, weight="weight")
            nx.write_graphml(mst, "graph.graphml")

            return False

        mst = nx.minimum_spanning_tree(self.G, weight="weight")
        nx.write_graphml(mst, "graph.graphml")
        return True
    


if __name__ == "__main__":

    # world = SamWorld("assets/challenge_video.mp4")
    world = SamWorld(
        "C:/Users/jleto/Downloads/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz"
    )

    try:
        while True:
            running = world.run()
            if not running:
                break

    except KeyboardInterrupt:
        print("Saving graph...")

    finally:
        mst = nx.minimum_spanning_tree(world.G, weight="weight")
        nx.write_graphml(mst, "graph.graphml")

        print("Graph saved.")


        pos = nx.kamada_kawai_layout(mst, weight="weight")

        edge_labels = nx.get_edge_attributes(mst, "weight")

        nx.draw(mst, pos, with_labels=True, node_size=1000)

        nx.draw_networkx_edge_labels(
            mst,
            pos,
            edge_labels=edge_labels
        )

        plt.show()