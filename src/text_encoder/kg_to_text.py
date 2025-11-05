# src/text_encoder/kg_to_text.py

import pickle
import os

def triples_to_text(graph_path, output_path):
    """
    Converts a Knowledge Graph into natural language sentences
    for LLM encoding.
    """
    # Load the Knowledge Graph
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    sentences = []

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "related to")
        sentence = f"{u} {rel} {v}."
        sentences.append(sentence)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"✅ Converted KG to text and saved at: {output_path}")

if __name__ == "__main__":
    graph_path = "data/knowledge_graph/kg_graph.gpickle"
    output_path = "data/knowledge_graph/kg_text.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    triples_to_text(graph_path, output_path)
