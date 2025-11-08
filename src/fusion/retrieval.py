# src/fusion/train_unified_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from src.fusion.projection_head import ProjectionModel


# =============================
# 1. Load Paired Embeddings
# =============================

IMG_PATH = "data/paired_embeddings/images.pt "
TXT_PATH = "data/paired_embeddings/texts.pt"


print("📥 Loading paired embeddings...")
try:
    img_embeds = torch.load(IMG_PATH, weights_only=False).float()
    txt_embeds = torch.load(TXT_PATH, weights_only=False).float()
except Exception as e:
    print(torch.load(IMG_PATH, weights_only=False))
    raise RuntimeError(f"❌ Error loading embeddings: {e}")
assert img_embeds.shape[0] == txt_embeds.shape[0], "❌ Image/Text count mismatch!"

print("✅ Loaded:")
print("🖼️ Image embeddings:", img_embeds.shape)
print("✍️ Text embeddings :", txt_embeds.shape)

N, IMG_DIM = img_embeds.shape
_, TXT_DIM = txt_embeds.shape

# Normalize before training (helps CLIP training)
img_embeds = F.normalize(img_embeds, dim=-1)
txt_embeds = F.normalize(txt_embeds, dim=-1)


# =============================
# 2. Dataset Wrapper
# =============================

class PairedDataset(Dataset):
    def __init__(self, imgs, txts):
        self.imgs = imgs
        self.txts = txts

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.txts[idx]


dataset = PairedDataset(img_embeds, txt_embeds)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)


# =============================
# 3. Initialize Model
# =============================

proj = ProjectionModel(
    img_dim=IMG_DIM,
    text_dim=TXT_DIM,
    hidden_dim=512
)

proj.train()

optimizer = torch.optim.AdamW(proj.parameters(), lr=1e-4, weight_decay=1e-4)
temperature = nn.Parameter(torch.tensor(0.07))   # CLIP temperature
EPOCHS = 100

print("\n🚀 Starting CLIP-style training...\n")


# =============================
# 4. CLIP Contrastive Loss
# =============================

def clip_contrastive_loss(img_z, txt_z, temp):
    """
    Computes symmetric CLIP loss:
        image->text and text->image
    """
    logits = img_z @ txt_z.T
    logits = logits / temp

    labels = torch.arange(len(logits))

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


# =============================
# 5. Training Loop
# =============================

for epoch in range(EPOCHS):
    total_loss = 0.0

    for img_batch, txt_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        # Project into shared space
        img_z, txt_z = proj(img_batch, txt_batch)

        # Normalize projections
        img_z = F.normalize(img_z, dim=-1)
        txt_z = F.normalize(txt_z, dim=-1)

        # Compute contrastive loss
        loss = clip_contrastive_loss(img_z, txt_z, temperature.exp())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"✅ Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")


# =============================
# 6. Save Model
# =============================

os.makedirs("data/models", exist_ok=True)
torch.save({
    "model_state": proj.state_dict(),
    "temperature": temperature.detach(),
}, "data/models/unified_projection_model.pt")

print("\n🎉 Training complete!")
print("✅ Saved final unified model → data/models/unified_projection_model.pt")
