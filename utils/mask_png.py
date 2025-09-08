import os
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

# ── Configure these ──────────────────────────────────────────────
INPUT_DIR  = r"cartocs\label"     # folder containing .tif/.tiff masks
OUTPUT_DIR = r"cartocs\label_png"     # will be created if missing
OUTPUT_SIZE = (256, 256)                      # (W, H)
# ────────────────────────────────────────────────────────────────

def read_mask_first_band(path):
    """Read first band as float64 array (H, W)."""
    with rasterio.open(path) as src:
        band1 = src.read(1, out_dtype=np.float64)  # 1-based index
    return band1

def to_binary01(arr):
    """
    Convert any mask array to strict binary {0,1} without stretching.
    Rules:
      - If integer dtype: any >0 -> 1, else 0
      - If float and max <= 1.0: threshold at 0.5
      - Else (floats with larger range): any >0 -> 1
    """
    a = arr
    if np.issubdtype(a.dtype, np.integer):
        b = (a > 0).astype(np.uint8)
    else:
        finite = np.isfinite(a)
        if not finite.any():
            b = np.zeros_like(a, dtype=np.uint8)
        else:
            maxv = np.nanmax(a)
            if maxv <= 1.0:
                b = (a >= 0.5).astype(np.uint8)
            else:
                b = (a > 0).astype(np.uint8)
    return b

def resize_nearest_uint8(mask01, size_wh):
    """Nearest-neighbor resize; keeps values in {0,1}."""
    W, H = size_wh
    img = Image.fromarray(mask01, mode="L")     # 0..255; our data are 0/1
    img = img.resize((W, H), Image.NEAREST)     # preserves binary values
    out = np.array(img, dtype=np.uint8)
    # re-enforce binary in case of any oddities
    out = (out > 0).astype(np.uint8)
    return out

def save_png_single_band(dst_path, mask01_uint8):
    """Write single-channel PNG (count=1) with dtype=uint8, values {0,1}."""
    profile = {
        "driver": "PNG",
        "height": mask01_uint8.shape[0],
        "width":  mask01_uint8.shape[1],
        "count": 1,
        "dtype": "uint8",
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(mask01_uint8, 1)

def process_one(tif_path, out_dir):
    try:
        m = read_mask_first_band(tif_path)
        m = np.where(np.isfinite(m), m, 0.0)
        m01 = to_binary01(m)                              # -> {0,1}
        m01r = resize_nearest_uint8(m01, OUTPUT_SIZE)     # -> 256x256 {0,1}
        out_path = Path(out_dir) / (Path(tif_path).stem + ".png")
        save_png_single_band(str(out_path), m01r)
        return True, str(out_path)
    except Exception as e:
        return False, f"{tif_path}: {e}"

def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = [p for ext in ("*.tif", "*.tiff") for p in in_dir.glob(ext)]
    if not tifs:
        print(f"No .tif/.tiff files found in {in_dir}")
        return

    print(f"Converting {len(tifs)} mask file(s) to {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} binary PNGs → {out_dir}")
    ok, fail = 0, 0
    for p in tqdm(tifs, unit="file"):
        success, msg = process_one(str(p), out_dir)
        if success:
            ok += 1
        else:
            fail += 1
            print("Failed:", msg)
    print(f"Done. Success: {ok}, Failed: {fail}")

if __name__ == "__main__":
    main()
