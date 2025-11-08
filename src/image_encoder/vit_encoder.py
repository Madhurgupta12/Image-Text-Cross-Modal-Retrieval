# src/image_encoder/vit_encoder.py

from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image
import os
from tqdm import tqdm

# Initialize ViT
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

def encode_image_folder(image_folder, output_path):
    """
    Encodes all images in a folder into embeddings using pretrained ViT.
    """
    embeddings = {}
    model.eval()

    with torch.no_grad():
        for img_file in tqdm(os.listdir(image_folder), desc="Encoding images"):
            img_path = os.path.join(image_folder, img_file)
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings[img_file] = embedding

    # Save all embeddings
    torch.save(embeddings, output_path)
    print(f"✅ Saved image embeddings to: {output_path}")

if __name__ == "__main__":
    
    DATASET_PATH = r"D:\UniProject\Image_Text_Cross_Modal_Retrieval\data\images\Images"
    output_path = "data/processed/image_embeddings/vit_embeddings.pt"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    encode_image_folder(DATASET_PATH, output_path)