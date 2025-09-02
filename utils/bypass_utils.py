# ========== Helpers ==========
def imread(path):
    """Read image as PIL (RGB)."""
    return Image.open(path).convert('RGB')

def imwrite(img_pil, path, fmt=None, **save_kwargs):
    """Save PIL image to disk."""
    path = str(path)
    if fmt is None:
        img_pil.save(path, **save_kwargs)
    else:
        img_pil.save(path, fmt, **save_kwargs)

def _ensure_odd(k):
    """Ensure kernel size is odd (required for GaussianBlur)."""
    k = int(k)
    return k if k % 2 == 1 else k + 1

# ========== Transformations ==========
def jpeg_compress(img_pil, quality=75):
    """JPEG recompression with adjustable quality."""
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def down_up_resize(img_pil, scale=0.5, resample=Image.BICUBIC):
    """Downscale then upscale image to introduce artifacts."""
    w, h = img_pil.size
    w2, h2 = max(1, int(w*scale)), max(1, int(h*scale))
    return img_pil.resize((w2, h2), resample=resample).resize((w, h), resample=resample)

def add_gaussian_noise(img_pil, std=10):
    """Add Gaussian noise with given std."""
    arr = np.array(img_pil).astype(np.int16)
    noise = np.random.normal(0, float(std), arr.shape).astype(np.int16)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def gaussian_blur(img_pil, ksize=5):
    """Apply Gaussian blur with kernel size ksize."""
    k = _ensure_odd(ksize)
    arr = np.array(img_pil)
    out = cv2.GaussianBlur(arr, (k, k), 0)
    return Image.fromarray(out)

def unsharp_mask(img_pil, amount=1.0, radius=2):
    """Unsharp mask (sharpening)."""
    src = np.array(img_pil).astype(np.float32)
    blur = cv2.GaussianBlur(src, (_ensure_odd(radius*2+1), _ensure_odd(radius*2+1)), 0)
    sharp = np.clip(src + float(amount)*(src - blur), 0, 255).astype(np.uint8)
    return Image.fromarray(sharp)

from PIL import ImageEnhance
def adjust_brightness(img_pil, factor=1.1):
    """Adjust brightness (factor >1 brighter, <1 darker)."""
    return ImageEnhance.Brightness(img_pil).enhance(float(factor))

def adjust_contrast(img_pil, factor=1.1):
    """Adjust contrast (factor >1 stronger, <1 weaker)."""
    return ImageEnhance.Contrast(img_pil).enhance(float(factor))

def overlay_patch(img_pil, patch_path=None, rel_x=0.6, rel_y=0.6, rel_w=0.15, rel_h=0.15, alpha=0.6):
    """
    Overlay an adversarial-style patch.
    rel_x/rel_y: top-left position (relative to width/height)
    rel_w/rel_h: relative patch size
    alpha: patch transparency (0-1)
    """
    base = img_pil.convert('RGBA')
    W, H = base.size
    w, h = max(1, int(W*rel_w)), max(1, int(H*rel_h))
    x, y = min(W-w, int(W*rel_x)), min(H-h, int(H*rel_y))

    if patch_path is not None:
        patch = Image.open(patch_path).convert('RGBA').resize((w, h))
    else:
        # generate random noise patch
        patch = Image.fromarray((np.random.rand(h, w, 3)*255).astype(np.uint8), 'RGB').convert('RGBA')

    # apply alpha
    a = Image.new('L', (w, h), color=int(255*float(alpha)))
    patch.putalpha(a)

    canvas = Image.new('RGBA', base.size)
    canvas.paste(patch, (x, y))
    out = Image.alpha_composite(base, canvas).convert('RGB')
    return out
