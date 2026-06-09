from openai import OpenAI
from utils.clip_embedding import embed_text
import numpy as np
import json
from dotenv import find_dotenv, load_dotenv
from pprint import pprint
from networkx.readwrite import json_graph
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")

SIM_THRESHOLD = 0.8

RAG = False

class ChatWithGraph:

    def __init__(self, graph):

        load_dotenv(override=True)

        self.client = OpenAI()

        self.graph = graph

        self.txt_embeddings = np.array([
            attrs["txt_embedding"]
            for _, attrs in graph.nodes(data=True)
        ])
        self.img_embeddings = np.array([
            attrs["img_embedding"]
            for _, attrs in graph.nodes(data=True)
        ])
        self.node_ids = [
            node_id
            for node_id, _ in graph.nodes(data=True)
        ]

        self.k = max(3, len(self.node_ids) // 4)

    def rag(self, prompt):
        doc = nlp(prompt)
        keywords = [
            token.text
            for token in doc
            if token.pos_ in [
                "NOUN",
                "PROPN",
                "VERB",
                "ADJ"
            ]
        ]

        print(keywords)

        # with open("graph.json") as f:
        #     graph = f.read()

        
        query_embedding = embed_text(" ".join(keywords))

        txt_score = cosine_similarity(
            [query_embedding],
            self.txt_embeddings
        )[0]
        img_score = cosine_similarity(
            [query_embedding],
            self.img_embeddings
        )[0]

        scores = 0.5 * txt_score + 0.5 * img_score


        # best_score = scores.max()
        # best_idx = np.where(
        #     scores >= best_score - 0.05
        # )[0]
        # best_idx = np.where(
        #     scores > SIM_THRESHOLD
        # )[0]
        # if len(best_idx) == 0:

        best_idx = np.argsort(scores)[-self.k:]

        top_nodes = [
            self.node_ids[i]
            for i in best_idx
        ]

        # for i in best_idx:
        #     print(
        #         f"{node_ids[i]}: {scores[i]:.4f}"
        #     )
        print("\nTop Retrieved:")

        for i in best_idx[::-1]:
            print(
                f"{self.node_ids[i]:15s} "
                f"txt={txt_score[i]:.4f} "
                f"img={img_score[i]:.4f} "
                f"final={scores[i]:.4f}"
            )

        return top_nodes

    def run(self):

        prompt = input("\nYou: ")

        if prompt.lower() in ["quit", "exit"]:
            return

        if RAG: 
            top_nodes = self.rag(prompt)

            expanded = set(top_nodes)
            for node in top_nodes:
                expanded.update(self.graph.neighbors(node))

            subgraph = self.graph.subgraph(expanded).copy()


        else: # use the whole graph with no rag
            subgraph = self.graph.copy()


        # remove embeddings attributes for llm
        for _, attrs in subgraph.nodes(data=True):
                attrs.pop("txt_embedding", None)
                attrs.pop("img_embedding", None)

        # in memory json
        graph_json = json.dumps(
                json_graph.node_link_data(subgraph),
                indent=2
            )

        # pprint(f"used graph: {graph_json}")

        # print("Retrieved nodes:")
        # for node in subgraph.nodes():
        #     print(node)

        response = self.client.responses.create(
            model="gpt-5",
            input=f"""
            You are a robot reasoning over a knowledge graph.

            Graph:
            {graph_json}

            User Question:
            {prompt}

            You are a robot reasoning over a spatial knowledge graph.

            Only recommend a destination if the user's request requires interaction with a physical object in the environment.

            If the request is unrelated to the environment, answer normally and do not select a destination.

            If there is insufficient information in the graph to answer confidently, say so.

            When a destination is required:
            1. Explain your reasoning.
            2. Select the best destination.
            3. Return the x, y, z coordinates.
            """
        )

        print("\nAssistant:")
        print(response.output_text)


