"""
Flask REST API for Image-Text Cross-Modal Retrieval.
Provides endpoints for text-to-image and image-to-image search.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import io
import pandas as pd

from src.fusion.projection_head import ProjectionModel
from src.text_encoder.llm_encoder import TextEncoder
from src.utils.config import PAIRED_EMB_DIR, MODELS_DIR, DATA_DIR


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for model and embeddings
model = None
text_encoder = None
image_projected = None
img_names = None
caption_df = None


def initialize_model():
    """Initialize model and load embeddings."""
    global model, text_encoder, image_projected, img_names, caption_df
    
    print("[*] Initializing retrieval system...")
    
    try:
        # Load model
        model_path = MODELS_DIR / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = ProjectionModel(img_dim=768, text_dim=384, shared_dim=512)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"[OK] Model loaded from {model_path}")
        
        # Load paired embeddings
        image_embeddings_path = PAIRED_EMB_DIR / "images.pt"
        img_names_path = PAIRED_EMB_DIR / "img_names.pt"
        
        if not image_embeddings_path.exists():
            raise FileNotFoundError(f"Image embeddings not found: {image_embeddings_path}")
        if not img_names_path.exists():
            raise FileNotFoundError(f"Image names not found: {img_names_path}")
        
        image_embeddings = torch.load(image_embeddings_path)
        img_names = torch.load(img_names_path)
        print(f"[OK] Loaded {len(image_embeddings)} image embeddings")
        
        # Project embeddings to shared space
        with torch.no_grad():
            image_projected = model.encode_image(image_embeddings, normalize=True)
        print(f"[OK] Projected embeddings to {image_projected.shape[1]}-dim shared space")
        print(f"[OK] Indexed {len(img_names)} unique images")
        
        # Initialize text encoder
        text_encoder = TextEncoder(model_name='sentence-transformers/all-MiniLM-L6-v2')
        print("[OK] Text encoder initialized")
        
        # Load captions
        global caption_df
        caption_df = pd.read_csv(DATA_DIR / "images" / "captions.txt")
        print(f"[OK] Loaded {len(caption_df)} captions")
        
        print("[OK] Retrieval system ready!")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_images': len(img_names) if img_names is not None else 0
    })


@app.route('/api/search/text', methods=['POST'])
def text_to_image_search():
    """Text-to-image search endpoint."""
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k <= 0:
            return jsonify({'error': 'top_k must be a positive integer'}), 400
        if top_k > 1000:
            return jsonify({'error': 'top_k cannot exceed 1000'}), 400
        
        # Encode and project query
        query_embedding = text_encoder.encode([query_text])
        with torch.no_grad():
            query_proj = model.encode_text(query_embedding, normalize=True)
        
        # Compute similarities
        similarities = torch.matmul(query_proj, image_projected.T)
        similarities = similarities.squeeze(0)
        
        # Get top-k results (fetch more to account for duplicates)
        fetch_k = min(top_k * 10, len(img_names))  # Fetch 10x more to ensure enough unique results
        top_scores, top_indices = torch.topk(similarities, k=fetch_k)
        
        # Format results - deduplicate by image name
        seen_images = set()
        results = []
        
        for score, idx in zip(top_scores.numpy(), top_indices.numpy()):
            img_name = img_names[idx]
            
            # Skip if we've already seen this image
            if img_name in seen_images:
                continue
            
            seen_images.add(img_name)
            img_path = str(DATA_DIR / "images" / "images" / img_name)
            
            results.append({
                'rank': len(results) + 1,
                'image_id': img_name,
                'image_path': img_path,
                'score': float(score)
            })
            
            # Stop once we have enough unique results
            if len(results) >= top_k:
                break
        
        return jsonify({
            'query': query_text,
            'num_results': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/image', methods=['POST'])
def image_to_image_search():
    """Image-to-image search endpoint."""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        image_index = data.get('image_index')
        top_k = data.get('top_k', 10)
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k <= 0:
            return jsonify({'error': 'top_k must be a positive integer'}), 400
        if top_k > 1000:
            return jsonify({'error': 'top_k cannot exceed 1000'}), 400
        
        # Find query image index
        if image_index is not None:
            query_idx = image_index
        elif image_id is not None:
            # Find index by image_id
            try:
                query_idx = list(img_names).index(image_id)
            except ValueError:
                return jsonify({'error': f'Image {image_id} not found'}), 404
        else:
            return jsonify({'error': 'Either image_id or image_index is required'}), 400
        
        if query_idx < 0 or query_idx >= len(img_names):
            return jsonify({'error': 'Invalid image index'}), 400
        
        # Get query embedding
        query_embedding = image_projected[query_idx].unsqueeze(0)
        
        # Compute similarities
        similarities = torch.matmul(query_embedding, image_projected.T)
        similarities = similarities.squeeze(0)
        
        # Get top-k results (fetch more to account for duplicates)
        fetch_k = min((top_k + 1) * 10, len(img_names))  # Fetch 10x more
        top_scores, top_indices = torch.topk(similarities, k=fetch_k)
        
        # Format results - deduplicate by image name (exclude query image itself)
        query_image_name = img_names[query_idx]
        seen_images = set([query_image_name])  # Start with query image in seen set
        results = []
        
        for score, idx in zip(top_scores.numpy(), top_indices.numpy()):
            img_name = img_names[idx]
            
            # Skip if we've already seen this image (includes query image)
            if img_name in seen_images:
                continue
            
            seen_images.add(img_name)
            img_path = str(DATA_DIR / "images" / "images" / img_name)
            
            results.append({
                'rank': len(results) + 1,
                'image_id': img_name,
                'image_path': img_path,
                'score': float(score)
            })
            
            # Stop once we have enough unique results
            if len(results) >= top_k:
                break
        
        return jsonify({
            'query_image': query_image_name,
            'query_index': query_idx,
            'num_results': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/image-to-text', methods=['POST'])
def image_to_text_search():
    """Image-to-text search endpoint - finds captions for an uploaded image."""
    try:
        # Check if image file is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        top_k = int(request.form.get('top_k', 5))
        
        # Validate inputs
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not isinstance(top_k, int) or top_k <= 0:
            return jsonify({'error': 'top_k must be a positive integer'}), 400
        if top_k > 100:
            return jsonify({'error': 'top_k cannot exceed 100'}), 400
        
        # Save uploaded image temporarily and process it
        from src.image_encoder.vit_encoder import ViTEncoder
        from torchvision import transforms
        
        # Read and preprocess the image
        image = Image.open(image_file.stream).convert('RGB')
        
        # Define preprocessing transforms (same as training)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocess image
        image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        
        # Encode the image
        image_encoder = ViTEncoder()
        with torch.no_grad():
            image_embedding = image_encoder.encode(image_tensor)  # [1, 768]
            
            # Project to shared space using forward method
            # We need to provide both image and text embeddings, use dummy text
            dummy_text = torch.zeros(1, 384)  # Create dummy text embedding same size as text_dim
            image_projected_query, _ = model.forward(image_embedding, dummy_text, normalize=True)  # [1, 512]
        
        # Calculate similarities with all text embeddings
        # We need to load text embeddings and find most similar captions
        text_embeddings_path = PAIRED_EMB_DIR / "texts.pt"
        text_embeddings = torch.load(text_embeddings_path)  # [N, 384]
        
        # Project text embeddings to shared space
        with torch.no_grad():
            dummy_images = torch.zeros(text_embeddings.shape[0], 768)  # Dummy images for batch
            _, text_projected = model.forward(dummy_images, text_embeddings, normalize=True)  # [N, 512]
        
        # Calculate similarities
        similarities = torch.nn.functional.cosine_similarity(
            image_projected_query, text_projected, dim=1
        )  # [N]
        
        # Get top-k similar captions
        top_scores, top_indices = torch.topk(similarities, min(top_k * 5, len(similarities)))
        
        # Get captions from caption_df using indices
        captions_list = []
        seen_captions = set()
        
        for idx in top_indices.numpy():
            caption = caption_df.iloc[idx]['caption'].strip()
            
            # Skip duplicates
            if caption in seen_captions:
                continue
            
            seen_captions.add(caption)
            captions_list.append(caption)
            
            if len(captions_list) >= top_k:
                break
        
        return jsonify({
            'num_captions': len(captions_list),
            'captions': captions_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<image_id>', methods=['GET'])
def get_image(image_id):
    """Serve image by ID."""
    try:
        image_path = DATA_DIR / "images" / "images" / image_id
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(str(image_path), mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<image_id>/thumbnail', methods=['GET'])
def get_thumbnail(image_id):
    """Serve thumbnail (resized) image."""
    try:
        size = int(request.args.get('size', 256))
        
        # Validate size parameter
        if size <= 0 or size > 2048:
            return jsonify({'error': 'Size must be between 1 and 2048'}), 400
        image_path = DATA_DIR / "images" / "images" / image_id
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        # Load and resize image
        img = Image.open(image_path)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Convert to bytes
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        return jsonify({
            'num_images': len(img_names) if img_names is not None else 0,
            'num_captions': len(caption_df) if caption_df is not None else 0,
            'embedding_dim': int(image_projected.shape[1]) if image_projected is not None else 0,
            'model_architecture': {
                'image_input_dim': 768,
                'text_input_dim': 384,
                'shared_dim': 512
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation."""
    return jsonify({
        'name': 'Image-Text Cross-Modal Retrieval API',
        'version': '1.0.0',
        'endpoints': {
            'health': 'GET /health',
            'text_search': 'POST /api/search/text',
            'image_search': 'POST /api/search/image',
            'image_to_text': 'POST /api/search/image-to-text',
            'get_image': 'GET /api/image/<image_id>',
            'get_thumbnail': 'GET /api/image/<image_id>/thumbnail',
            'stats': 'GET /api/stats'
        }
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  Image-Text Cross-Modal Retrieval API")
    print("=" * 60)
    
    # Initialize model before starting server
    initialize_model()
    
    print("\n" + "=" * 60)
    print("  Starting Flask Server")
    print("=" * 60)
    print("\nAPI Endpoints:")
    print("  - Health Check:      http://localhost:5000/health")
    print("  - Text Search:       POST http://localhost:5000/api/search/text")
    print("  - Image Search:      POST http://localhost:5000/api/search/image")
    print("  - Image to Text:     POST http://localhost:5000/api/search/image-to-text")
    print("  - Get Image:         GET http://localhost:5000/api/image/<id>")
    print("  - Get Thumbnail:     GET http://localhost:5000/api/image/<id>/thumbnail")
    print("  - System Stats:      GET http://localhost:5000/api/stats")
    print("\n" + "=" * 60)
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=False)
