import os
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

# ── Configure these ──────────────────────────────────────────────
INPUT_DIR = r"cartosat_data_12\B"                      # folder containing .tif/.tiff
OUTPUT_DIR = r"cartosat_data_12\B_png" # will be created if missing
RGB_BAND_INDEXES = (3, 2, 1)                # 1-based band indices you asked for
OUTPUT_SIZE = (256, 256)                    # (W, H)
OUTPUT_BIT_DEPTH = 16                       # 8 or 16
CLIP_PERCENTILES = (2, 98)                  # per-band contrast stretch
TRY_ALTERNATE_ORDER_IF_WEIRD = False        # set True to auto-try (4,3,2) if image looks off
# ────────────────────────────────────────────────────────────────

def read_rgb(path, rgb_idx):
    with rasterio.open(path) as src:
        count = src.count
        if max(rgb_idx) <= count:
            bands = [src.read(i, out_dtype=np.float64) for i in rgb_idx]
        elif count >= 3:
            bands = [src.read(i, out_dtype=np.float64) for i in (1, 2, 3)]
        else:
            raise ValueError(f"{path} has only {count} band(s); need at least 3.")
        arr = np.stack(bands, axis=0)  # (3, H, W)
    return arr

def per_band_to_uint(arr):
    """
    Per-band percentile stretch to uint8/uint16.
    This avoids one channel dominating and producing a color cast.
    """
    out = np.empty_like(arr, dtype=np.uint16 if OUTPUT_BIT_DEPTH == 16 else np.uint8)
    for c in range(arr.shape[0]):
        band = arr[c]
        finite = np.isfinite(band)
        if not finite.any():
            band = np.zeros_like(band, dtype=np.float64)
            finite = np.ones_like(band, dtype=bool)

        lo, hi = np.percentile(band[finite], CLIP_PERCENTILES)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(band), np.nanmax(band)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
                band = np.zeros_like(band, dtype=np.float64)

        norm = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        if OUTPUT_BIT_DEPTH == 8:
            out[c] = (norm * 255.0 + 0.5).astype(np.uint8)
        else:
            out[c] = (norm * 65535.0 + 0.5).astype(np.uint16)
    return out

def resize_to(rgb_u, size_wh):
    """Resize to (W,H) per band using PIL; preserves uint8/uint16."""
    W, H = size_wh
    bands, h, w = rgb_u.shape
    mode = "I;16" if rgb_u.dtype == np.uint16 else "L"
    out = []
    for i in range(bands):
        img = Image.fromarray(rgb_u[i], mode=mode)
        img = img.resize((W, H), Image.BILINEAR)
        out.append(np.array(img))
    return np.stack(out, axis=0)  # (3,H,W)

def save_png_with_rasterio(dst_path, rgb_u):
    """Use GDAL/Rasterio to write 8/16-bit RGB PNG reliably."""
    profile = {
        "driver": "PNG",
        "height": rgb_u.shape[1],
        "width":  rgb_u.shape[2],
        "count": 3,
        "dtype": rgb_u.dtype,
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(rgb_u)

def maybe_try_alternate_order(rgb_idx):
    # Common natural-color for Sentinel-2 is (4,3,2). If your current order looks odd,
    # you can flip this flag at the top OR just change RGB_BAND_INDEXES directly.
    return (4, 3, 2) if rgb_idx != (4, 3, 2) else (3, 2, 1)

def process_one(tif_path, out_dir):
    try:
        # First pass with requested bands
        rgb = read_rgb(tif_path, RGB_BAND_INDEXES)
        rgb = np.where(np.isfinite(rgb), rgb, 0.0)
        rgb_u = per_band_to_uint(rgb)
        rgb_resized = resize_to(rgb_u, OUTPUT_SIZE)

        # Optional quick heuristic: if the image is severely red-tinted, try alternate order
        if TRY_ALTERNATE_ORDER_IF_WEIRD:
            r_mean = rgb_resized[0].mean()
            g_mean = rgb_resized[1].mean()
            b_mean = rgb_resized[2].mean()
            if r_mean > 1.6 * max(g_mean, b_mean):  # crude check for red dominance
                alt_idx = maybe_try_alternate_order(RGB_BAND_INDEXES)
                rgb_alt = read_rgb(tif_path, alt_idx)
                rgb_alt = np.where(np.isfinite(rgb_alt), rgb_alt, 0.0)
                rgb_alt_u = per_band_to_uint(rgb_alt)
                rgb_resized = resize_to(rgb_alt_u, OUTPUT_SIZE)

        out_path = Path(out_dir) / (Path(tif_path).stem + ".png")
        save_png_with_rasterio(str(out_path), rgb_resized)
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

    print(f"Converting {len(tifs)} file(s) to {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} PNGs → {out_dir}")
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
