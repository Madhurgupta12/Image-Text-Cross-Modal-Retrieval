"""
Utility script to download Flickr30k dataset from Kaggle.
Requires kagglehub to be installed and Kaggle API credentials configured.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

try:
    import kagglehub
except ImportError:
    print("❌ kagglehub not installed. Install with: pip install kagglehub")
    exit(1)


def download_flickr30k(target_dir: Optional[str] = None) -> str:
    """
    Download Flickr30k dataset from Kaggle.
    
    Args:
        target_dir: Target directory to save dataset (default: data/images)
    
    Returns:
        Path to downloaded dataset
    """
    # Set target directory
    if target_dir is None:
        target_dir = os.path.join("data", "images")
    
    print("📥 Downloading Flickr30k dataset from Kaggle...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("adityajn105/flickr30k")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy dataset files
        print(f"📦 Copying dataset to: {target_dir}")
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(target_dir, item)
            
            if os.path.isdir(s):
                if os.path.exists(d):
                    print(f"  ⚠️  Directory {item} already exists, skipping...")
                else:
                    shutil.copytree(s, d, dirs_exist_ok=True)
                    print(f"  ✓ Copied directory: {item}")
            else:
                if os.path.exists(d):
                    print(f"  ⚠️  File {item} already exists, skipping...")
                else:
                    shutil.copy2(s, d)
                    print(f"  ✓ Copied file: {item}")
        
        print(f"\n✅ Dataset successfully copied to: {target_dir}")
        return target_dir
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nPlease ensure:")
        print("  1. kagglehub is installed: pip install kagglehub")
        print("  2. Kaggle API credentials are configured")
        print("  3. You have accepted the dataset terms on Kaggle")
        raise


if __name__ == "__main__":
    download_flickr30k()
