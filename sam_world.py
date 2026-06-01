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
from scripts.frame_dif import frame_dif
from scripts.clip_embedding import embed
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

load_dotenv(override=True)

# LLM Setup
client = OpenAI()

# SAM3 Setup
overrides = dict(
    conf=0.8,
    task="segment",
    mode="predict",
    model="sam3.pt",
)
sam_predictor = SAM3SemanticPredictor(overrides=overrides)

# Graph Display Setup
plt.ion()
plt.figure(figsize=(6, 6))
plt.show(block=False)
G = nx.Graph()
pos = {}

# Video Setup
cap = cv2.VideoCapture("assets/challenge_video.mp4")

# JSON-Based Graph Setup
GRAPH_PATH = "graph.json"
graph = {
    "nodes": [],
    "edges": []
}
with open(GRAPH_PATH, "w") as f:
    json.dump(graph, f, indent=2)   


embedding_matrix = []
metadata = []
def add_node(node, embedding, node_id):
    graph["nodes"].append(node)
    embedding_matrix.append(embedding)
    metadata.append(node_id)
    G.add_node(node_id)
    if len(G.nodes) > 1: G.add_edge(node_id, random.choice(list(G.nodes)[:-1]))
    print("NEW OBJECT !!!!!!!!!!!!!!!!!!!!!!!")


frame_count = 0

SAM_STEP = 5
CHANGE_STEP = 5

SIM_THRESHOLD = .8

DEFAULT_LABELS = ["road", "car", "tree"]

run_gpt = False
prev_frame = None
sam_labels = []

while True:
    ret, frame = cap.read()

    if not ret: break

    frame_count += 1

    # Only run SAM every SAM_STEP frames
    if frame_count % SAM_STEP != 0: 
        continue

    # Stepped Frame Differencing Logic
    if prev_frame is None:
        prev_frame = frame
        run_gpt = True
        score = None
    elif frame_count % CHANGE_STEP == 0:
        run_gpt = frame_dif(prev_frame, frame)

    # Prepare image for LLM
    frame  = cv2.resize(frame, (640, 360))
    _, buffer = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    
    # Run LLM if significant change detected or first frame
    if run_gpt:
        print("--- LLM RUNNING ---")
        sam_labels = llm(client, base64_image, DEFAULT_LABELS)
        print(sam_labels)

    full_labels = DEFAULT_LABELS + sam_labels # Combine default and LLM labels for SAM3

    # Run SAM3 with the combined labels
    results = sam_predictor(
        frame,
        text=full_labels,
        # imgsz=448,
        save=False,
        verbose=False
    )
    result = results[0]
    annotated = result.plot() # Annotate the frame with SAM3 results

    # Add detected labels to graph
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        node_id = f"{label}_{len(graph['nodes'])}"

        mask = result.masks.data[i]
        embedding = embed(frame, mask)

        # JSON-Based Graph Update
        node = {
            "id": node_id,
            "label": label,
            "embedding": embedding,
        }

        if len(embedding_matrix) > 0:
            sims = cosine_similarity(
                [embedding],
                embedding_matrix
            )[0]
            best_sim = sims.max()

            print(best_sim)

            if best_sim < SIM_THRESHOLD:
                add_node(node, embedding, node_id)
            else: 
                # merging logic
                pass
        else:
            add_node(node, embedding, node_id)

        with open(GRAPH_PATH, "w") as f:
            json.dump(graph, f, indent=2)

        # NetworkX Graph Update
        

    # Display graph using NetworkX and Matplotlib
    plt.clf()
    if len(pos) == 0:
        pos = nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G, pos=pos)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        font_size=8
    )
    plt.pause(0.1)
    plt.draw()
    # plt.tight_layout()

    # Display the annotated frame
    cv2.imshow("SAM3 Video", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
