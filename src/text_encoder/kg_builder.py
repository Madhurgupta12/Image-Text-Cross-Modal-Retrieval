# src/text_encoder/kg_builder.py

import networkx as nx
from tqdm import tqdm
import os
import spacy
import pickle

# Load spaCy model for NLP-based relation extraction
nlp = spacy.load("en_core_web_sm")

def extract_entities_and_relations(text):
    """
    Uses spaCy to extract (subject, relation, object) triples from text.
    Example: 'A man riding a bike' → ('man', 'riding', 'bike')
    """
    doc = nlp(text)
    triples = []
    for token in doc:
        if token.dep_ in ("ROOT", "advcl", "xcomp") and token.pos_ == "VERB":
            subject = None
            object_ = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child.text
                elif child.dep_ in ("dobj", "attr", "pobj"):
                    object_ = child.text
            if subject and object_:
                triples.append((subject, token.lemma_, object_))
    return triples


def build_knowledge_graph(captions_file, output_path):
    """
    Build a simple Knowledge Graph from captions text file.
    Each caption line contributes entity–relation–object triples.
    """
    print("🔍 Building Knowledge Graph...")
    G = nx.DiGraph()

    # Read captions file
    with open(captions_file, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f if line.strip()]

    for caption in tqdm(captions, desc="Processing captions"):
        triples = extract_entities_and_relations(caption)
        for subj, rel, obj in triples:
            G.add_edge(subj, obj, relation=rel)

    # Save graph using pickle
    with open(output_path, "wb") as f:
        pickle.dump(G, f)

    print(f"✅ Knowledge Graph saved at: {output_path}")
    return G


if __name__ == "__main__":
    captions_path = "data/captions/train_captions.txt"  # ✅ your actual captions file
    output_path = "data/knowledge_graph/kg_graph.gpickle"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    build_knowledge_graph(captions_path, output_path)
