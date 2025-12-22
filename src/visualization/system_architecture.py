import networkx as nx
import matplotlib.pyplot as plt


def draw_system_architecture():
    G = nx.DiGraph()

    nodes = [
        "Real Data",
        "Synthetic Data",
        "NLP Documents",
        "training_dataset",
        "Feature Engineering",
        "ML Models",
        "MLflow Tracking"
    ]

    edges = [
        ("Real Data", "training_dataset"),
        ("Synthetic Data", "training_dataset"),
        ("NLP Documents", "training_dataset"),
        ("training_dataset", "Feature Engineering"),
        ("Feature Engineering", "ML Models"),
        ("ML Models", "MLflow Tracking")
    ]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue")
    plt.title("System Architecture Diagram")
    plt.show()


if __name__ == "__main__":
    draw_system_architecture()
