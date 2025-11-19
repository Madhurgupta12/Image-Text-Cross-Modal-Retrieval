# Image-Text Cross-Modal Retrieval System

A comprehensive multimodal representation learning framework for image-text retrieval using Vision Transformers (ViT) and Large Language Models (LLMs). This system encodes paired image and text data into a unified embedding space, enabling efficient cross-modal retrieval with REST API deployment and web interface.

## 🌟 Features

- **Vision Transformer (ViT) Encoding**: Google  ViT-base-patch16-224 with batch processing and GPU acceleration
- **LLM Text Encoding**: Sentence-transformers/all-MiniLM-L6-v2 for efficient text embeddings
- **Unified Embedding Space**: 512-dimensional shared space with learnable projection heads
- **CLIP-Style Training**: Contrastive learning with learnable temperature parameter (τ = 0.07)
- **Comprehensive Evaluation**: Recall@K (1, 5, 10), alignment & uniformity metrics
- **REST API**: Flask-based API with CORS support for production deployment
- **Web Interface**: Interactive HTML/CSS/JavaScript UI for real-time retrieval
- **Visualization Tools**: Generate query result visualizations and embedding plots
- **Production-Ready**: Full training pipeline with checkpointing, early stopping, and comprehensive logging

## 📋 Requirements

- Python 3.8+
- PyTorch 2.9.0+
- CUDA 11.8+ (for GPU support, optional - CPU training supported)
- Flask 3.1.2+ (for API deployment)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Madhurgupta12/Image-Text-Cross-Modal-Retrieval.git
cd Image-Text-Cross-Modal-Retrieval
```

### 2. Create a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# API dependencies (if deploying REST API)
pip install -r requirements_api.txt
```

## 📁 Project Structure

```
Image-Text-Cross-Modal-Retrieval/
├── src/
│   ├── image_encoder/
│   │   ├── vit_encoder.py          # Vision Transformer encoder (ViT-base)
│   │   ├── dataset_loader.py       # Dataset loading and preprocessing
│   │   ├── preprocess.py           # Image preprocessing utilities
│   │   └── save_embeddings.py      # Save image embeddings
│   ├── text_encoder/
│   │   ├── llm_encoder.py          # LLM text encoder (all-MiniLM-L6-v2)
│   │   └── text_cleaning.py        # Text preprocessing
│   ├── fusion/
│   │   ├── projection_head.py      # Projection networks (768→512, 384→512)
│   │   ├── pair_embeddings.py      # Embedding pairing utilities
│   │   ├── retrieval.py            # Loss functions & retrieval
│   │   └── train_projection.py     # Projection training logic
│   ├── evaluation/
│   │   ├── metrics.py              # Evaluation metrics (Recall@K)
│   │   ├── visualize_embeddings.py # Embedding visualization (t-SNE/UMAP)
│   │   └── qualitative_results.py  # Qualitative analysis
│   └── utils/
│       ├── config.py               # Configuration management
│       ├── seed_all.py             # Reproducibility utilities
│       ├── logging_utils.py        # Logging infrastructure
│       ├── helpers.py              # Helper functions
│       └── download_dataset.py     # Dataset download utilities
├── data/
│   ├── images/
│   │   └── images/                 # Flickr30k image files (8,091 images)
│   ├── results.csv                 # Image-caption pairs (40,455 pairs)
│   ├── paired_embeddings/          # Projected embeddings
│   │   ├── images.pt               # Image embeddings (512-dim)
│   │   ├── texts.pt                # Text embeddings (512-dim)
│   │   └── img_names.pt            # Image filenames
│   └── processed/                  # Raw embeddings
├── models/
│   └── best_model.pt               # Trained projection model (1.12M params)
├── outputs/
│   ├── logs/                       # Training logs
│   ├── results/                    # Evaluation results
│   └── visualizations/             # Generated visualizations (6 PNG files)
├── train_pipeline.py               # Main training script
├── inference.py                    # Inference and retrieval
├── demo_visualizations.py          # Generate visualization demos
├── api_server.py                   # Flask REST API server
├── web_interface.html              # Interactive web UI
├── requirements.txt                # Core dependencies
├── requirements_api.txt            # API dependencies
└── README.md                       # This file
```

## 🎯 Quick Start

### Option 1: Use Pre-trained Model (Recommended)

If you have a trained model (`models/best_model.pt`) and embeddings ready:

```bash
# Start the API server
python api_server.py

# Open web interface in browser
# Open web_interface.html in your browser
# or
start web_interface.html  # Windows
open web_interface.html   # Mac
```

The API will be available at `http://localhost:5000`

### Option 2: Train from Scratch

#### 1. Prepare Your Dataset

Place your image-text paired dataset in the following structure:
```
data/
├── images/
│   └── images/              # Image files
└── results.csv              # Image-caption pairs (columns: image, caption)
```

The `results.csv` should have columns: `image` (filename) and `caption` (text description).

**Supported Datasets:**
- **Flickr30k**: 8,091 images with 40,455 captions (5 per image)
- **MS-COCO**: 330,000 images with multiple captions
- **Custom datasets**: Any image-caption paired dataset

#### 2. Train the Complete Pipeline

```bash
# This will:
# 1. Extract image embeddings using ViT
# 2. Extract text embeddings using MiniLM
# 3. Create paired embeddings
# 4. Train projection heads (50 epochs)
# 5. Evaluate on validation set
python train_pipeline.py
```

**Training Output:**
- Model checkpoints saved to `models/`
- Best model: `models/best_model.pt`
- Paired embeddings: `data/paired_embeddings/`
- Training logs: `outputs/logs/`

#### 3. Generate Visualizations

```bash
# Generate demo visualizations for 5 text queries
python demo_visualizations.py
```

**Output:** 6 PNG files in `outputs/visualizations/`:
- `query_1.png` to `query_5.png` - Top-5 retrieval results per query
- `top_matches_grid.png` - Grid view of all results

#### 4. Deploy API and Web Interface

```bash
# Start Flask API server
python api_server.py

# In another terminal/browser, open:
# web_interface.html
```
```

## 🌐 REST API Usage

### API Endpoints

The Flask API provides 6 endpoints for cross-modal retrieval:

#### 1. Health Check
```bash
GET http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_images": 8091
}
```

#### 2. Text-to-Image Search
```bash
POST http://localhost:5000/api/search/text
Content-Type: application/json

{
  "query": "a dog playing in the park",
  "top_k": 10
}
```

**Response:**
```json
{
  "query": "a dog playing in the park",
  "num_results": 10,
  "results": [
    {
      "rank": 1,
      "image_id": "image_001.jpg",
      "image_path": "data/images/images/image_001.jpg",
      "score": 0.8756
    },
    ...
  ]
}
```

#### 3. Image-to-Image Search
```bash
POST http://localhost:5000/api/search/image
Content-Type: application/json

{
  "image_id": "image_001.jpg",
  "top_k": 10
}
```

#### 4. Get Image
```bash
GET http://localhost:5000/api/image/<image_id>
```

Returns the full-resolution image file.

#### 5. Get Thumbnail
```bash
GET http://localhost:5000/api/image/<image_id>/thumbnail?size=256
```

Returns a resized thumbnail (default: 256x256).

#### 6. System Statistics
```bash
GET http://localhost:5000/api/stats
```

**Response:**
```json
{
  "num_images": 8091,
  "num_captions": 40455,
  "embedding_dim": 512,
  "model_architecture": {
    "image_input_dim": 768,
    "text_input_dim": 384,
    "shared_dim": 512
  }
}
```

### Python Client Example

```python
import requests

# Text search
response = requests.post(
    'http://localhost:5000/api/search/text',
    json={'query': 'sunset over mountains', 'top_k': 5}
)
results = response.json()

# Display results
for item in results['results']:
    print(f"Rank {item['rank']}: {item['image_id']} (score: {item['score']:.4f})")
```

### Web Interface Features

Open `web_interface.html` in your browser to access:

- **Real-time Search**: Type queries and see results instantly
- **Example Queries**: Pre-populated example searches
- **Grid View**: Visual display of top-K results with scores
- **Image Preview**: Click to view full-resolution images
- **Statistics Dashboard**: System info and dataset stats
- **Responsive Design**: Works on desktop and mobile browsers

## 🔧 Configuration

### Model Configuration

Edit `src/utils/config.py` to customize settings:


Edit `src/utils/config.py` to customize settings:

```python
# src/utils/config.py

class ModelConfig:
    # Vision Transformer
    VIT_MODEL_NAME = "google/vit-base-patch16-224"
    VIT_OUTPUT_DIM = 768
    
    # Text Encoder
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_OUTPUT_DIM = 384
    
    # Projection Head
    SHARED_DIM = 512
    PROJECTION_HIDDEN_DIMS = [512]
    USE_BATCH_NORM = True
    USE_DROPOUT = True
    DROPOUT_RATE = 0.1
```

### Training Configuration

```python
class TrainingConfig:
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    OPTIMIZER = "adamw"
    SCHEDULER_TYPE = "cosine"
    LOSS_TYPE = "clip"  # Options: 'clip', 'triplet', 'infonce'
    
    USE_EARLY_STOPPING = True
    PATIENCE = 10
    
    # CLIP loss temperature
    TEMPERATURE = 0.07
    LEARNABLE_TEMPERATURE = True
```

## 📊 Model Performance

### Trained on Flickr30k Dataset

**Dataset Statistics:**
- Total images: 8,091
- Total captions: 40,455 (5 per image)
- Train/Val split: 80/20

**Model Architecture:**
- Image encoder: ViT-base-patch16-224 (86M params, frozen)
- Text encoder: all-MiniLM-L6-v2 (23M params, frozen)
- Projection heads: 1.12M trainable parameters
- Shared embedding space: 512 dimensions

**Retrieval Performance (Best Epoch: 48/50):**

| Metric | Image→Text | Text→Image |
|--------|------------|------------|
| Recall@1 | 47.90% | 44.69% |
| Recall@5 | 74.38% | 71.10% |
| Recall@10 | 83.52% | 80.75% |

**Training Details:**
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Loss: CLIP Contrastive Loss with learnable temperature
- Scheduler: Cosine annealing
- Batch size: 128
- Training time: ~2 hours on CPU
- Best model saved at epoch 48

## 📊 Evaluation

## 📊 Evaluation

Evaluate your trained model:

```python
from src.evaluation.metrics import evaluate_bidirectional_retrieval, print_metrics

# Load embeddings
image_embeddings = torch.load('data/paired_embeddings/images.pt')
text_embeddings = torch.load('data/paired_embeddings/texts.pt')

# Evaluate retrieval performance
metrics = evaluate_bidirectional_retrieval(
    image_embeddings, 
    text_embeddings,
    k_values=[1, 5, 10]
)

print_metrics(metrics, "Retrieval Performance")
```

**Metrics Explained:**
- **Recall@K**: Percentage of queries where the correct match appears in top-K results
- **Median Rank**: Median position of the correct match in ranking
- **Mean Rank**: Average position of the correct match

## 📈 Visualization

### Generate Demo Visualizations

```bash
python demo_visualizations.py
```

This generates visualizations for 5 text queries:
1. "people on the beach"
2. "a child climbing stairs"
3. "a dog running through water"
4. "city skyline at night"
5. "mountain landscape with lake"

**Output:** 6 PNG files showing top-5 retrieval results for each query.

### Custom Visualizations

```python
from src.evaluation.visualize_embeddings import visualize_embeddings

# Visualize embeddings using t-SNE or UMAP
visualize_embeddings(
    image_embeddings, 
    text_embeddings,
    method='umap',  # or 'tsne'
    save_path='outputs/visualizations/embeddings.png'
)
```

## 🔬 Advanced Features

### Command-Line Inference

```python
from inference import load_retriever
import torch

# Load model and embeddings
model = torch.load('models/best_model.pt')
image_embeddings = torch.load('data/paired_embeddings/images.pt')
img_names = torch.load('data/paired_embeddings/img_names.pt')

# Create retriever
from src.fusion.retrieval import CrossModalRetriever
retriever = CrossModalRetriever(model, image_embeddings, img_names)

# Text-to-image search
query = "sunset over mountains"
results = retriever.search_by_text(query, top_k=5)

for rank, (img_name, score) in enumerate(results, 1):
    print(f"{rank}. {img_name} (score: {score:.4f})")
```

### Programmatic API Usage

```python
import requests

# Start API server first: python api_server.py

# Text search
def text_search(query, top_k=10):
    response = requests.post(
        'http://localhost:5000/api/search/text',
        json={'query': query, 'top_k': top_k}
    )
    return response.json()

# Image search
def image_search(image_id, top_k=10):
    response = requests.post(
        'http://localhost:5000/api/search/image',
        json={'image_id': image_id, 'top_k': top_k}
    )
    return response.json()

# Usage
results = text_search("a cat sitting on a couch", top_k=5)
for item in results['results']:
    print(f"Rank {item['rank']}: {item['image_id']} ({item['score']:.4f})")
```

## 🛠️ Troubleshooting

### Out of Memory Error
- **Solution 1**: Reduce `BATCH_SIZE` in `src/utils/config.py` (try 64 or 32)
- **Solution 2**: Use CPU instead of GPU: set `DEVICE = 'cpu'`
- **Solution 3**: Process images in smaller batches

### API Server Won't Start
- **Check port**: Ensure port 5000 is not already in use
- **Check files**: Verify `models/best_model.pt` and `data/paired_embeddings/` exist
- **Check dependencies**: Run `pip install -r requirements_api.txt`

### Poor Retrieval Performance
- **Solution 1**: Train for more epochs (increase `NUM_EPOCHS`)
- **Solution 2**: Increase `SHARED_DIM` (try 768 or 1024)
- **Solution 3**: Use larger batch size if memory allows
- **Solution 4**: Fine-tune temperature parameter

### Web Interface Not Loading
- **Check API**: Ensure `api_server.py` is running on port 5000
- **Check CORS**: API has CORS enabled by default
- **Check browser console**: Look for JavaScript errors

### Slow Training
- **Solution 1**: Reduce validation frequency (set `VAL_EVERY_N_EPOCHS = 5`)
- **Solution 2**: Use GPU if available
- **Solution 3**: Reduce `BATCH_SIZE` to speed up batches
- **Solution 4**: Freeze encoder weights (already done by default)

### Missing Files Error
```
FileNotFoundError: Model file not found
```
- **Solution**: Train the model first using `python train_pipeline.py`
- Or ensure `models/best_model.pt` exists

## 🚀 Deployment

### Production Deployment

For production deployment, consider:

1. **Use Gunicorn** (Linux/Mac):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

2. **Use Waitress** (Windows):
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 api_server:app
```

3. **Enable HTTPS**:
- Use nginx as reverse proxy
- Configure SSL certificates

4. **Add Authentication**:
- Implement API key validation
- Use JWT tokens for secure access

5. **Add Rate Limiting**:
```bash
pip install flask-limiter
```

6. **Monitor Performance**:
- Log response times
- Monitor memory usage
- Track query patterns

### Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt requirements_api.txt ./
RUN pip install -r requirements.txt -r requirements_api.txt

COPY . .

EXPOSE 5000
CMD ["python", "api_server.py"]
```

Build and run:
```bash
docker build -t image-text-retrieval .
docker run -p 5000:5000 image-text-retrieval
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{image-text-retrieval-2025,
  title={Image-Text Cross-Modal Retrieval System},
  author={Madhur Gupta},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/Madhurgupta12/Image-Text-Cross-Modal-Retrieval}}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome from all, Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  Acknowledgments

- [CLIP](https://github.com/openai/CLIP) - Inspiration for contrastive learning approach
- [Sentence Transformers](https://www.sbert.net/) - Text encoding models
- [Hugging Face Transformers](https://huggingface.co/transformers/) - ViT models and model hub
- [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/) - Dataset for training and evaluation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework for API deployment

## 📧 Contact

**Madhur Gupta**
- GitHub: [@Madhurgupta12](https://github.com/Madhurgupta12)
- Repository: [Image-Text-Cross-Modal-Retrieval](https://github.com/Madhurgupta12/Image-Text-Cross-Modal-Retrieval)

For questions or issues, please open an issue on GitHub.

## 🎓 Project Statistics

- **Total Lines of Code**: ~5,000+
- **Python Files**: 25+
- **Dependencies**: 15+ core packages
- **Model Parameters**: 1.12M trainable (110M total with frozen encoders)
- **Training Time**: ~2 hours on CPU (50 epochs)
- **Dataset**: Flickr30k (8,091 images, 40,455 captions)
- **API Endpoints**: 6 RESTful endpoints
- **Supported Operations**: Text→Image, Image→Image retrieval

## ⭐ Key Features Summary

✅ **CLIP-style contrastive learning** with learnable temperature  
✅ **Dual projection heads** (image: 768→512, text: 384→512)  
✅ **Production-ready REST API** with Flask + CORS  
✅ **Interactive web interface** with real-time search  
✅ **Comprehensive evaluation** (Recall@1/5/10)  
✅ **Visualization tools** for qualitative analysis  
✅ **Checkpointing & early stopping** for optimal training  
✅ **CPU and GPU support** for flexible deployment  
✅ **Well-documented** with inline comments and README  
✅ **Bug-free code** with proper error handling  

---

**Built with ❤️ for multimodal AI research and applications**


