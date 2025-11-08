import torch
import os

IMG_PATH = "data/processed/image_embeddings/vit_embeddings.pt"
TXT_PATH = "data/processed/text_embeddings/text_embeds.pt"

OUT_IMG = "data/paired_embeddings/images.pt"
OUT_TXT = "data/paired_embeddings/texts.pt"
OUT_NAMES = "data/paired_embeddings/img_names.pt"

print("📥 Loading image embeddings...")
img_embeds = torch.load(IMG_PATH, weights_only=False)

print("📥 Loading text embeddings...")
txt = torch.load(TXT_PATH, weights_only=False)

# ---- Step 1: Extract image vectors + filenames ----
if isinstance(img_embeds, dict):
    img_names = list(img_embeds.keys())
    img_vectors = [torch.tensor(v).float() for v in img_embeds.values()]
else:
    raise ValueError("❌ Expected image embeddings as dict {filename: vector}")

# ---- Step 2: Extract caption embeddings ----
if "embeddings" in txt:
    txt_vectors = torch.tensor(txt["embeddings"]).float()
else:
    raise ValueError("❌ Expected text_embeds.pt in format {captions, image_names, embeddings}")

# ---- Step 3: Pair 1 image ↔ 5-caption average ----
paired_img = []
paired_txt = []

print("🔗 Pairing images with their 5 captions...")

for i in range(len(img_vectors)):
    img_vec = img_vectors[i]

    start = i * 5
    end = start + 5

    if end > len(txt_vectors):
        break

    cap5 = txt_vectors[start:end]
    avg_caption = torch.mean(cap5, dim=0)

    paired_img.append(img_vec)
    paired_txt.append(avg_caption)

paired_img = torch.stack(paired_img)
paired_txt = torch.stack(paired_txt)

# ---- Step 4: Save ----
os.makedirs("data/paired_embeddings", exist_ok=True)

torch.save(paired_img, OUT_IMG)
torch.save(paired_txt, OUT_TXT)
torch.save(img_names, OUT_NAMES)

print("\n✅ Pairing complete!")
print(f"🖼️ Images saved:  {OUT_IMG}   shape={paired_img.shape}")
print(f"✍️ Text saved:    {OUT_TXT}   shape={paired_txt.shape}")
print(f"📛 Filenames saved at: {OUT_NAMES}")
