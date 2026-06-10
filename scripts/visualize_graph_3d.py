import argparse
import json
from pathlib import Path

from networkx.readwrite import json_graph


def get_pyplot(output=None):
    try:
        import matplotlib

        if output:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to visualize graphs. "
            "Install it in this environment or run from your project env."
        ) from exc

    return plt


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    radius = 0.5 * max(x_range, y_range, z_range)

    x_mid = sum(x_limits) / 2
    y_mid = sum(y_limits) / 2
    z_mid = sum(z_limits) / 2

    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def load_graph(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    return json_graph.node_link_graph(data)


def draw_graph_3d(graph, title=None, output=None):
    plt = get_pyplot(output)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for node, attrs in graph.nodes(data=True):
        if not all(key in attrs for key in ("world_x", "world_y", "world_z")):
            continue

        x = attrs["world_x"]
        y = attrs["world_y"]
        z = attrs["world_z"]
        cluster = attrs.get("cluster", -1)
        color = plt.cm.tab10(cluster % 10)

        ax.scatter(x, y, z, color=color, s=120)
        ax.text(x, y, z, str(node), fontsize=9)

    for u, v, attrs in graph.edges(data=True):
        if u not in graph.nodes or v not in graph.nodes:
            continue

        a = graph.nodes[u]
        b = graph.nodes[v]
        if not all(key in a and key in b for key in ("world_x", "world_y", "world_z")):
            continue

        x1, y1, z1 = a["world_x"], a["world_y"], a["world_z"]
        x2, y2, z2 = b["world_x"], b["world_y"], b["world_z"]

        ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", alpha=0.45)

        if "weight" in attrs:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                (z1 + z2) / 2,
                f"{attrs['weight']:.2f}",
                fontsize=8,
            )

    ax.set_xlabel("world_x")
    ax.set_ylabel("world_y")
    ax.set_zlabel("world_z")
    if title:
        ax.set_title(title)
    set_axes_equal(ax)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=200)
        print(f"Wrote 3D graph image to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a NetworkX node-link JSON graph in 3D."
    )
    parser.add_argument("graph_json", help="Path to graph JSON file")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional image path. If omitted, opens an interactive plot.",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph_json)
    graph = load_graph(graph_path)
    draw_graph_3d(graph, title=graph_path.name, output=args.output)


if __name__ == "__main__":
    main()
