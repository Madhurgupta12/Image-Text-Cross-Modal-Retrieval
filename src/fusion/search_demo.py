# src/fusion/search_demo.py

import torch
import torch.nn.functional as F
from src.fusion.projection_head import ProjectionModel

# ============================================================
# 1. Paths
# ============================================================

IMG_PATH = "data/processed/image_embeddings/vit_embeddings.pt"
TXT_PATH = "data/processed/text_embeddings/text_embeds.pt"
MODEL_PATH = "data/models/unified_projection_model.pt"   # IMPORTANT

# ============================================================
# 2. Load embeddings
# ============================================================

print("📥 Loading image embeddings...")
img_embeds = torch.load(IMG_PATH, weights_only=False)

print("📥 Loading text embeddings...")
txt_data = torch.load(TXT_PATH, weights_only=False)

captions = txt_data["captions"]             # list of captions
caption_img_names = txt_data["image_names"] # list of original images per caption
txt_embeds = txt_data["embeddings"]         # text embeddings tensor


# --- Convert image dict to stacked tensor ---
if isinstance(img_embeds, dict):
    img_names = list(img_embeds.keys())
    img_embeds = torch.stack([torch.tensor(v).float() for v in img_embeds.values()])
else:
    img_names = list(range(len(img_embeds)))

txt_embeds = txt_embeds.float()

print("✅ Embedding shapes:")
print("Images:", img_embeds.shape)
print("Texts :", txt_embeds.shape)

# ============================================================
# 3. Load Projection Model
# ============================================================

print("📥 Loading trained projection model...")
checkpoint = torch.load(MODEL_PATH, weights_only=False)

proj = ProjectionModel(
    img_dim=img_embeds.shape[1],
    text_dim=txt_embeds.shape[1],
    hidden_dim=512
)

proj.load_state_dict(checkpoint["model_state"])
temperature = checkpoint["temperature"].exp()
proj.eval()

print("✅ Model loaded.")


# ============================================================
# 4. Project both embeddings
# ============================================================

print("🔧 Projecting all embeddings...")

with torch.no_grad():
    img_proj = proj.image_proj(img_embeds)
    txt_proj = proj.text_proj(txt_embeds)

# Normalize embeddings
img_proj = F.normalize(img_proj, dim=-1)
txt_proj = F.normalize(txt_proj, dim=-1)

print("✅ Projection complete.")


# ============================================================
# 5. Helper: Show real caption text
# ============================================================

def show_query_text(idx):
    print("\n============================")
    print(f"📌 TEXT QUERY [{idx}]")
    print("Caption:", captions[idx])
    print("Original caption image:", caption_img_names[idx])
    print("============================\n")


# ============================================================
# 6. Retrieval Function
# ============================================================

def retrieve(text_idx, top_k=5):
    """Retrieve top-K matching images for a text query."""
    show_query_text(text_idx)

    query = txt_proj[text_idx]
    sims = torch.matmul(img_proj, query)

    topk = torch.topk(sims, top_k)

    print(f"✅ Top {top_k} matching images for text index {text_idx}:\n")
    for rank, img_idx in enumerate(topk.indices.tolist()):
        print(f"{rank+1}. {img_names[img_idx]}   score={topk.values[rank]:.4f}")


# ============================================================
# 7. Run retrieval demo
# ============================================================

if __name__ == "__main__":
    print("\n🚀 Running retrieval demo...\n")
    for i in range(5):   # show first 5 captions
        retrieve(i, top_k=5)
