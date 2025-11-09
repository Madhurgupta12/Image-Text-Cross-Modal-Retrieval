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


def initialize_model():
    """Initialize model and load embeddings."""
    global model, text_encoder, image_projected, img_names
    
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
        
        # Get top-k results
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(img_names)))
        
        # Format results
        results = []
        for rank, (score, idx) in enumerate(zip(top_scores.numpy(), top_indices.numpy()), 1):
            img_name = img_names[idx]
            img_path = str(DATA_DIR / "images" / "images" / img_name)
            
            results.append({
                'rank': rank,
                'image_id': img_name,
                'image_path': img_path,
                'score': float(score)
            })
        
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
        
        # Get top-k results
        top_scores, top_indices = torch.topk(similarities, k=min(top_k + 1, len(img_names)))
        
        # Format results (exclude query image itself)
        results = []
        for rank, (score, idx) in enumerate(zip(top_scores.numpy(), top_indices.numpy())):
            if idx == query_idx:
                continue
            
            img_name = img_names[idx]
            img_path = str(DATA_DIR / "images" / "images" / img_name)
            
            results.append({
                'rank': len(results) + 1,
                'image_id': img_name,
                'image_path': img_path,
                'score': float(score)
            })
            
            if len(results) >= top_k:
                break
        
        query_image_name = img_names[query_idx]
        
        return jsonify({
            'query_image': query_image_name,
            'query_index': query_idx,
            'num_results': len(results),
            'results': results
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
        # Load caption data for stats
        caption_df = pd.read_csv(DATA_DIR / "images" / "results.csv")
        
        return jsonify({
            'num_images': len(img_names) if img_names is not None else 0,
            'num_captions': len(caption_df),
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
    print("  - Get Image:         GET http://localhost:5000/api/image/<id>")
    print("  - Get Thumbnail:     GET http://localhost:5000/api/image/<id>/thumbnail")
    print("  - System Stats:      GET http://localhost:5000/api/stats")
    print("\n" + "=" * 60)
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=False)
