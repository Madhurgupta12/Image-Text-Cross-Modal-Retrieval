
import os
DATASET_PATH = r"C:\Users\hp\.cache\kagglehub\datasets\adityajn105\flickr30k\versions\1"
# Paths for dataset
IMAGES_DIR = os.path.join("data", "images", "images")
CAPTIONS_FILE = os.path.join("data", "images", "results.csv")

# Processed paths
CAPTION_TXT_PATH = os.path.join("data", "captions", "train_captions.txt")
KG_GRAPH_PATH = os.path.join("data", "knowledge_graph", "kg_graph.gpickle")
KG_TEXT_PATH = os.path.join("data", "knowledge_graph", "kg_text.txt")

# Embedding save paths
IMAGE_EMB_PATH = os.path.join("data", "processed", "image_embeddings", "vit_embeddings.pt")
TEXT_EMB_PATH = os.path.join("data", "processed", "text_embeddings", "kg_text_embeddings.pt")
