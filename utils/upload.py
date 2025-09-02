# ==== Basic setup ====
!pip -q install pillow opencv-python numpy tqdm

import io, os, json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from IPython.display import display

# Define folders
ROOT = Path('/content')
UPLOAD_DIR  = ROOT/'uploads'
PATCHED_DIR = ROOT/'uploads_bypass'
LOG_DIR     = ROOT/'logs'
for p in [UPLOAD_DIR, PATCHED_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder):
    """List all image files in a folder (common formats)."""
    exts = {'.png','.jpg','.jpeg','.webp','.bmp'}
    return sorted([p for p in Path(folder).glob('**/*') if p.suffix.lower() in exts])

print("Directories created:")
print("UPLOAD_DIR :", UPLOAD_DIR)
print("PATCHED_DIR:", PATCHED_DIR)
print("LOG_DIR    :", LOG_DIR)
