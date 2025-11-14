"""
Text encoder using Language Models for caption embedding.
Supports multiple models including Sentence Transformers, BERT, RoBERTa, etc.
"""

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from src.utils.config import (
    ModelConfig, DeviceConfig, TEXT_EMB_PATH, CAPTIONS_FILE, DataConfig
)


class TextEncoder(nn.Module):
    """
    Text encoder wrapper supporting multiple model types.
    """
    
    def __init__(
        self,
        model_name: str = ModelConfig.TEXT_MODEL_NAME,
        device: Optional[torch.device] = None,
        use_sentence_transformer: bool = True,
        pooling: str = "mean"  # Options: mean, cls, max
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            use_sentence_transformer: Use SentenceTransformer (easier) or raw transformers
            pooling: Pooling strategy for token embeddings
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or DeviceConfig.DEVICE
        self.use_sentence_transformer = use_sentence_transformer
        self.pooling = pooling
        
        print(f"🔧 Loading text encoder: {model_name}")
        
        try:
            if use_sentence_transformer:
                self.model = SentenceTransformer(model_name)
                self.model.to(self.device)
                self.output_dim = self.model.get_sentence_embedding_dimension()
                self.tokenizer = None
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.output_dim = self.model.config.hidden_size
            
            self.model.eval()
            
            print(f"✅ Text encoder loaded successfully")
            print(f"   Output dimension: {self.output_dim}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load text encoder: {e}")
    
    def _preprocess_text(self, texts: List[str]) -> List[str]:
        """Preprocess text captions"""
        processed = []
        for text in texts:
            # Convert to string
            text = str(text).strip()
            
            # Apply preprocessing
            if DataConfig.LOWERCASE:
                text = text.lower()
            
            if DataConfig.REMOVE_PUNCTUATION:
                import string
                text = text.translate(str.maketrans('', '', string.punctuation))
            
            processed.append(text)
        
        return processed
    
    def _pool_token_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token embeddings into sentence embedding.
        
        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Sentence embeddings [batch_size, hidden_dim]
        """
        if self.pooling == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling == "cls":
            return token_embeddings[:, 0, :]  # CLS token
        
        elif self.pooling == "max":
            # Max pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[mask_expanded == 0] = -1e9  # Set padding to large negative
            return torch.max(token_embeddings, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar
            max_length: Maximum sequence length
        
        Returns:
            Text embeddings [N, D]
        """
        self.model.eval()
        
        # Preprocess texts
        texts = self._preprocess_text(texts)
        max_length = max_length or DataConfig.MAX_TEXT_LENGTH
        
        if self.use_sentence_transformer:
            # Use SentenceTransformer's encode method
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=normalize
            )
            return embeddings.cpu()
        
        else:
            # Manual encoding with raw transformers
            embeddings = []
            
            iterator = range(0, len(texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding texts")
            
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**encoded)
                
                # Pool embeddings
                batch_embeddings = self._pool_token_embeddings(
                    outputs.last_hidden_state,
                    encoded['attention_mask']
                )
                
                embeddings.append(batch_embeddings.cpu())
            
            embeddings = torch.cat(embeddings, dim=0)
            
            # Normalize
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
            return embeddings
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for use in end-to-end training.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Text embeddings [batch_size, D]
        """
        if self.use_sentence_transformer:
            raise NotImplementedError("Forward pass not supported with SentenceTransformer. Use encode() instead.")
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool_token_embeddings(outputs.last_hidden_state, attention_mask)


def load_captions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load image-caption pairs from CSV file.
    
    Args:
        csv_path: Path to captions CSV file
    
    Returns:
        DataFrame with 'image' and 'caption' columns
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Captions file not found: {csv_path}")
    
    try:
        # Try different separators
        for sep in [',', '|', '\t']:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                if len(df.columns) >= 2:
                    break
            except (pd.errors.ParserError, UnicodeDecodeError) as e:
                continue
        
        # Ensure we have the right columns
        if 'image' not in df.columns:
            # Assume first column is image, second is caption
            df.columns = ['image', 'caption'] + list(df.columns[2:])
        
        # Clean data
        df['caption'] = df['caption'].astype(str).str.strip()
        df['image'] = df['image'].astype(str).str.strip()
        
        # Remove invalid entries
        df = df[df['caption'].str.len() > 0]
        df = df[df['image'].str.len() > 0]
        
        print(f"📖 Loaded {len(df)} captions from {len(df['image'].unique())} images")
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"Error loading captions file: {e}")


def encode_captions(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    model_name: str = ModelConfig.TEXT_MODEL_NAME,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    normalize: bool = True,
    save_metadata: bool = True
):
    """
    Encode captions and save embeddings.
    
    Args:
        df: DataFrame with 'image' and 'caption' columns
        output_path: Path to save embeddings
        model_name: Text encoder model name
        batch_size: Batch size for encoding
        device: Device to use
        normalize: Whether to normalize embeddings
        save_metadata: Whether to save captions and image names
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize encoder
    encoder = TextEncoder(model_name=model_name, device=device)
    
    # Get captions
    captions = df['caption'].tolist()
    image_names = df['image'].tolist()
    
    print(f"✍️ Encoding {len(captions)} captions...")
    
    # Encode
    embeddings = encoder.encode(
        captions,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=True
    )
    
    # Save
    save_data = {
        'embeddings': embeddings,
    }
    
    if save_metadata:
        save_data['captions'] = captions
        save_data['image_names'] = image_names
    
    torch.save(save_data, output_path)
    
    print(f"✅ Saved text embeddings to: {output_path}")
    print(f"   Shape: {embeddings.shape}")


def encode_caption_list(
    captions: List[str],
    model_name: str = ModelConfig.TEXT_MODEL_NAME,
    device: Optional[torch.device] = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Encode a list of captions.
    
    Args:
        captions: List of caption strings
        model_name: Text encoder model name
        device: Device to use
        normalize: Whether to normalize
    
    Returns:
        Text embeddings [N, D]
    """
    encoder = TextEncoder(model_name=model_name, device=device)
    return encoder.encode(captions, normalize=normalize, show_progress=False)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Encode captions using LLM")
    parser.add_argument(
        "--captions_file",
        type=str,
        default=str(CAPTIONS_FILE),
        help="Path to captions CSV file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(TEXT_EMB_PATH),
        help="Path to save embeddings"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=ModelConfig.TEXT_MODEL_NAME,
        help="Text encoder model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    
    args = parser.parse_args()
    
    # Load captions
    try:
        df = load_captions(args.captions_file)
        
        # Encode captions
        encode_captions(
            df=df,
            output_path=args.output_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            normalize=True
        )
        
        print(f"\n🎉 Successfully encoded {len(df)} captions!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
