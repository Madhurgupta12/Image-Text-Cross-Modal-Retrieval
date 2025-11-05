# src/text_encoder/kg_inspect.py

import pickle
import networkx as nx
import matplotlib.pyplot as plt

GRAPH_PATH = "data/knowledge_graph/kg_graph.gpickle"
TEXT_PATH = "data/knowledge_graph/kg_text.txt"

def inspect_knowledge_graph(graph_path):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print("🧠 Number of nodes:", len(G.nodes()))
    print("🔗 Number of edges:", len(G.edges()))

    # Print a few sample edges
    print("\n📚 Sample relations:")
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        if i == 5:  # print only first 5 edges
            break
        rel = data.get("relation", "related to")
        print(f"  • {u} --({rel})--> {v}")

    # Quick visualization (optional)
    plt.figure(figsize=(8, 6))
    subgraph = list(G.edges())[:15]  # show first 15 edges
    H = G.edge_subgraph(subgraph)
    pos = nx.spring_layout(H)
    nx.draw(H, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1500, font_size=10)
    edge_labels = nx.get_edge_attributes(H, 'relation')
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)
    plt.title("🔍 Sample Knowledge Graph (subset)")
    plt.show()

def preview_kg_text(text_path):
    print("\n📝 Preview of KG-Text sentences:")
    with open(text_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 5:  # show first 5 sentences
                break
            print(f"  {i+1}. {line.strip()}")

if __name__ == "__main__":
    inspect_knowledge_graph(GRAPH_PATH)
    preview_kg_text(TEXT_PATH)
