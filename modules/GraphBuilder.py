import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from networkx.readwrite import json_graph
import json

class GraphBuilder:
    def __init__(self):
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.G = nx.Graph()
        self.pos = {}

    # Adds to Graph and updates memories
    def add_object(self, obj, global_objects):

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

        for other_obj in global_objects:
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

    def _cluster(self, final_graph):
        nodes = list(final_graph.nodes())

        X = np.array([
            [
                final_graph.nodes[n]["world_x"],
                final_graph.nodes[n]["world_y"],
                final_graph.nodes[n]["world_z"],
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
            final_graph.nodes[node]["cluster"] = int(cluster_id)

        return final_graph

    def _build_topology(self):
        threshold_graph = nx.Graph(
            (u, v, d)
            for u, v, d in self.G.edges(data=True)
            if d["weight"] < 0.5
        )

        mst = nx.minimum_spanning_tree(self.G, weight="weight")

        final_graph = nx.compose(mst, threshold_graph)

        final_graph = self._cluster(final_graph)

        return final_graph

    def save_graph(self, filename="graph.json"):
        data = json_graph.node_link_data(self._build_topology())
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def draw_2d_graph(self):

        plt.clf()

        final_graph = self._build_topology()

        edge_labels = nx.get_edge_attributes(final_graph, "weight")

        node_colors = [
            final_graph.nodes[n]["cluster"]
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

        self.save_graph()

        plt.pause(0.1)
        plt.draw()

        return final_graph
    
    def draw_3d_graph(self):
        plt.close("all")


        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        final_graph = self._build_topology()

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

        self.save_graph()

        plt.show(block=False)

        return final_graph
    
    
    