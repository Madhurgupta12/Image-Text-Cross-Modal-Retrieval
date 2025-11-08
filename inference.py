"""
Inference system for image-text cross-modal retrieval.
Supports efficient similarity search and bi-directional retrieval.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image

from src.fusion.projection_head import ProjectionModel
from src.image_encoder.vit_encoder import ViTEncoder
from src.text_encoder.llm_encoder import TextEncoder
from src.utils.config import (
    ModelConfig, DeviceConfig, BEST_MODEL_PATH, PROJECTION_MODEL_PATH,
    IMAGE_EMB_PATH, TEXT_EMB_PATH
)


class CrossModalRetriever:
    """
    Cross-modal retrieval system for image-text matching.
    """
    
    def __init__(
        self,
        projection_model_path: Union[str, Path],
        image_encoder: Optional[ViTEncoder] = None,
        text_encoder: Optional[TextEncoder] = None,
        device: Optional[torch.device] = None,
        use_faiss: bool = False
    ):
        """
        Args:
            projection_model_path: Path to trained projection model
            image_encoder: Pre-initialized image encoder (optional)
            text_encoder: Pre-initialized text encoder (optional)
            device: Device to run on
            use_faiss: Whether to use FAISS for efficient search
        """
        self.device = device or DeviceConfig.DEVICE
        self.use_faiss = use_faiss
        
        # Load projection model
        print(f"🔧 Loading projection model from {projection_model_path}")
        self.projection_model = ProjectionModel()
        checkpoint = torch.load(projection_model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.projection_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.projection_model.load_state_dict(checkpoint)
        
        self.projection_model.to(self.device)
        self.projection_model.eval()
        
        # Initialize encoders
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # FAISS index
        self.faiss_index = None
        self.image_embeddings = None
        self.text_embeddings = None
        self.image_ids = None
        self.text_ids = None
        
        print("✅ Retriever initialized")
    
    def _init_image_encoder(self):
        """Lazy initialization of image encoder"""
        if self.image_encoder is None:
            print("🔧 Initializing image encoder...")
            self.image_encoder = ViTEncoder(
                model_name=ModelConfig.VIT_MODEL_NAME,
                device=self.device
            )
    
    def _init_text_encoder(self):
        """Lazy initialization of text encoder"""
        if self.text_encoder is None:
            print("🔧 Initializing text encoder...")
            self.text_encoder = TextEncoder(
                model_name=ModelConfig.TEXT_MODEL_NAME,
                device=self.device
            )
    
    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List[Image.Image], List[str], torch.Tensor],
        batch_size: int = 32,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode images into the shared embedding space.
        
        Args:
            images: List of PIL Images, image paths, or pre-computed embeddings
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        
        Returns:
            Image embeddings in shared space [N, D]
        """
        self._init_image_encoder()
        
        # Load images if paths provided
        if isinstance(images[0], (str, Path)):
            images = [Image.open(img).convert('RGB') for img in images]
        
        # Encode with ViT
        if isinstance(images[0], Image.Image):
            vit_embeddings = self.image_encoder.encode(
                images, batch_size=batch_size, normalize=True, show_progress=False
            )
        else:
            vit_embeddings = images
        
        # Move to device
        vit_embeddings = vit_embeddings.to(self.device)
        
        # Project to shared space
        shared_embeddings = self.projection_model.encode_image(vit_embeddings, normalize=normalize)
        
        return shared_embeddings.cpu()
    
    @torch.no_grad()
    def encode_texts(
        self,
        texts: Union[List[str], torch.Tensor],
        batch_size: int = 64,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode texts into the shared embedding space.
        
        Args:
            texts: List of text strings or pre-computed embeddings
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        
        Returns:
            Text embeddings in shared space [N, D]
        """
        self._init_text_encoder()
        
        # Encode with LLM
        if isinstance(texts, list) and isinstance(texts[0], str):
            llm_embeddings = self.text_encoder.encode(
                texts, batch_size=batch_size, normalize=True, show_progress=False
            )
        else:
            llm_embeddings = texts
        
        # Move to device
        llm_embeddings = llm_embeddings.to(self.device)
        
        # Project to shared space
        shared_embeddings = self.projection_model.encode_text(llm_embeddings, normalize=normalize)
        
        return shared_embeddings.cpu()
    
    def build_index(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_ids: Optional[List] = None,
        text_ids: Optional[List] = None
    ):
        """
        Build search index from pre-computed embeddings.
        
        Args:
            image_embeddings: Image embeddings [N, D]
            text_embeddings: Text embeddings [M, D]
            image_ids: Image identifiers
            text_ids: Text identifiers
        """
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.image_ids = image_ids or list(range(len(image_embeddings)))
        self.text_ids = text_ids or list(range(len(text_embeddings)))
        
        # Build FAISS index if requested
        if self.use_faiss:
            try:
                import faiss
                
                # Normalize embeddings
                img_emb_np = F.normalize(image_embeddings, p=2, dim=-1).numpy()
                txt_emb_np = F.normalize(text_embeddings, p=2, dim=-1).numpy()
                
                # Create index
                dim = img_emb_np.shape[1]
                self.image_faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
                self.text_faiss_index = faiss.IndexFlatIP(dim)
                
                self.image_faiss_index.add(img_emb_np)
                self.text_faiss_index.add(txt_emb_np)
                
                print("✅ FAISS index built")
                
            except ImportError:
                print("⚠️ FAISS not available, using PyTorch for search")
                self.use_faiss = False
        
        print(f"✅ Index built with {len(self.image_ids)} images and {len(self.text_ids)} texts")
    
    def search_by_text(
        self,
        query_texts: Union[str, List[str]],
        top_k: int = 10,
        return_scores: bool = True
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """
        Search for images given text queries.
        
        Args:
            query_texts: Single text or list of texts
            top_k: Number of results to return
            return_scores: Whether to return similarity scores
        
        Returns:
            List of image indices (and optionally scores)
        """
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        
        # Encode queries
        query_embeddings = self.encode_texts(query_texts)
        
        if self.use_faiss and hasattr(self, 'image_faiss_index'):
            # FAISS search
            query_np = F.normalize(query_embeddings, p=2, dim=-1).numpy()
            scores, indices = self.image_faiss_index.search(query_np, top_k)
            
            results = []
            for i in range(len(query_texts)):
                result_indices = indices[i].tolist()
                result_scores = scores[i].tolist() if return_scores else None
                results.append((result_indices, result_scores) if return_scores else result_indices)
            
            return results[0] if len(query_texts) == 1 else results
        
        else:
            # PyTorch search
            similarities = query_embeddings @ self.image_embeddings.T
            scores, indices = similarities.topk(top_k, dim=1)
            
            results = []
            for i in range(len(query_texts)):
                result_indices = indices[i].tolist()
                result_scores = scores[i].tolist() if return_scores else None
                results.append((result_indices, result_scores) if return_scores else result_indices)
            
            return results[0] if len(query_texts) == 1 else results
    
    def search_by_image(
        self,
        query_images: Union[Image.Image, List[Image.Image], List[str]],
        top_k: int = 10,
        return_scores: bool = True
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """
        Search for texts given image queries.
        
        Args:
            query_images: Single image or list of images/paths
            top_k: Number of results to return
            return_scores: Whether to return similarity scores
        
        Returns:
            List of text indices (and optionally scores)
        """
        if not isinstance(query_images, list):
            query_images = [query_images]
        
        # Encode queries
        query_embeddings = self.encode_images(query_images)
        
        if self.use_faiss and hasattr(self, 'text_faiss_index'):
            # FAISS search
            query_np = F.normalize(query_embeddings, p=2, dim=-1).numpy()
            scores, indices = self.text_faiss_index.search(query_np, top_k)
            
            results = []
            for i in range(len(query_images)):
                result_indices = indices[i].tolist()
                result_scores = scores[i].tolist() if return_scores else None
                results.append((result_indices, result_scores) if return_scores else result_indices)
            
            return results[0] if len(query_images) == 1 else results
        
        else:
            # PyTorch search
            similarities = query_embeddings @ self.text_embeddings.T
            scores, indices = similarities.topk(top_k, dim=1)
            
            results = []
            for i in range(len(query_images)):
                result_indices = indices[i].tolist()
                result_scores = scores[i].tolist() if return_scores else None
                results.append((result_indices, result_scores) if return_scores else result_indices)
            
            return results[0] if len(query_images) == 1 else results
    
    def get_image_id(self, index: int):
        """Get image ID from index"""
        return self.image_ids[index] if self.image_ids else index
    
    def get_text_id(self, index: int):
        """Get text ID from index"""
        return self.text_ids[index] if self.text_ids else index


def load_retriever(
    model_path: Union[str, Path] = BEST_MODEL_PATH,
    device: Optional[torch.device] = None,
    use_faiss: bool = False
) -> CrossModalRetriever:
    """
    Convenience function to load a retriever.
    
    Args:
        model_path: Path to projection model
        device: Device to use
        use_faiss: Whether to use FAISS
    
    Returns:
        CrossModalRetriever instance
    """
    return CrossModalRetriever(
        projection_model_path=model_path,
        device=device,
        use_faiss=use_faiss
    )


# Example usage
if __name__ == "__main__":
    print("🔍 Testing inference system...")
    
    # Load retriever
    retriever = load_retriever()
    
    # Test with dummy data
    print("\n1. Creating dummy embeddings...")
    dummy_img_emb = torch.randn(100, ModelConfig.SHARED_DIM)
    dummy_txt_emb = torch.randn(100, ModelConfig.SHARED_DIM)
    
    retriever.build_index(dummy_img_emb, dummy_txt_emb)
    
    # Test text-to-image search
    print("\n2. Testing text-to-image search...")
    query_embeddings = torch.randn(1, ModelConfig.SHARED_DIM)
    indices, scores = retriever.search_by_text(
        ["A dog playing in the park"],
        top_k=5,
        return_scores=True
    )
    print(f"   Top 5 indices: {indices}")
    print(f"   Scores: {[f'{s:.4f}' for s in scores]}")
    
    print("\n✅ Inference system test complete!")
