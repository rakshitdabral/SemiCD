import os
import shutil
import cv2
import numpy as np
import rasterio

# ----------------------------
# Alignment helpers
# ----------------------------
def align_feature_homography(ref_gray, mov_gray, max_features=5000, keep_ratio=0.3):
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(mov_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError("No descriptors found.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * keep_ratio)]

    if len(good) < 4:
        raise RuntimeError("Not enough good matches for homography.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return H, kp1, kp2, good, mask

def safe_homography(ref_gray, mov_gray, min_inliers=10, max_scale=4.0):
    try:
        H, kp1, kp2, matches, mask = align_feature_homography(ref_gray, mov_gray)
        inliers = mask.sum() if mask is not None else 0

        if inliers < min_inliers:
            print(f"Too few inliers ({inliers}), using identity.")
            return np.eye(3), kp1, kp2, matches, mask, False

        det = np.linalg.det(H[0:2, 0:2])
        if det < 1/max_scale or det > max_scale:
            print(f"Suspicious scaling (det={det:.2f}), using identity.")
            return np.eye(3), kp1, kp2, matches, mask, False

        return H, kp1, kp2, matches, mask, True

    except Exception as e:
        print(f"Alignment failed ({e}), using identity.")
        return np.eye(3), [], [], [], None, False

# ----------------------------
# Readers
# ----------------------------
def read_as_grayscale(path, method="composite"):
    with rasterio.open(path) as src:
        img = src.read()  # (bands, H, W) or (channels, H, W) for PNGs
        if img.shape[0] > 1:  # multiband
            img = np.transpose(img, (1, 2, 0))
        else:  # grayscale PNG
            return cv2.normalize(img[0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img = img.astype(np.float32)

    if img.shape[2] >= 4:  # B,G,R,NIR
        weights = np.array([0.1, 0.4, 0.4, 0.1], dtype=np.float32)
        gray = np.tensordot(img, weights[:img.shape[2]], axes=([2], [0]))
    else:  # fallback to green or middle channel
        gray = img[:, :, min(1, img.shape[2]-1)]

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray

def read_fcc(path):
    with rasterio.open(path) as src:
        img = src.read()
        if img.shape[0] >= 3:
            img = np.transpose(img[:3], (1, 2, 0))  # take first 3 bands
        else:
            img = np.stack([img[0]]*3, axis=-1)
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

# ----------------------------
# Alignment runner
# ----------------------------
def run_alignment(ref_path, mov_path, method="composite"):
    ref_gray = read_as_grayscale(ref_path, method)
    mov_gray = read_as_grayscale(mov_path, method)

    ref_fcc = read_fcc(ref_path)
    mov_fcc = read_fcc(mov_path)

    H, kp1, kp2, matches, mask, ok = safe_homography(ref_gray, mov_gray)

    aligned_fcc = cv2.warpPerspective(mov_fcc, H, (ref_fcc.shape[1], ref_fcc.shape[0]))

    valid_mask = (aligned_fcc.sum(axis=2) > 0).astype(np.uint8)
    valid_mask = np.repeat(valid_mask[:, :, None], 3, axis=2)

    ref_fcc_masked = ref_fcc * valid_mask
    aligned_fcc_masked = aligned_fcc * valid_mask
    mov_fcc_masked = mov_fcc * valid_mask

    return ref_fcc_masked, mov_fcc_masked, aligned_fcc_masked, kp1, kp2, matches, mask, ok

# ----------------------------
# Process list of pairs
# ----------------------------
def process_from_list(list_file, dir_A, dir_B, out_root):
    with open(list_file, "r") as f:
        file_names = [line.strip() for line in f if line.strip()]

    out_A = os.path.join(out_root, "A")
    out_B = os.path.join(out_root, "B")
    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)

    for i, fname in enumerate(file_names, 1):
        ref_path = os.path.join(dir_A, fname)
        mov_path = os.path.join(dir_B, fname)

        ref_out = os.path.join(out_A, fname)
        mov_out = os.path.join(out_B, fname)

        # Skip if already processed
        if os.path.exists(ref_out) and os.path.exists(mov_out):
            print(f"[{i}/{len(file_names)}] Skipping {fname}, already exists.")
            continue

        if not os.path.exists(ref_path) or not os.path.exists(mov_path):
            print(f"[{i}/{len(file_names)}] Missing: {ref_path} or {mov_path}, skipping...")
            continue

        try:
            ref, mov, aligned, _, _, _, _, ok = run_alignment(ref_path, mov_path, method="composite")

            if not ok:
                print(f"[{i}/{len(file_names)}] Alignment failed for {fname}, copying originals...")
                shutil.copy(ref_path, ref_out)
                shutil.copy(mov_path, mov_out)
                continue

            cv2.imwrite(ref_out, cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mov_out, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))

            print(f"[{i}/{len(file_names)}] Saved aligned pair: {fname}")

        except Exception as e:
            print(f"[{i}/{len(file_names)}] Error processing {fname}: {e}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    list_file = r"D:\Rakshit\SemiCD\Cartosat\list\100_train_unsupervised.txt"
    dir_A = r"D:\Rakshit\SemiCD\Cartosat\A"
    dir_B = r"D:\Rakshit\SemiCD\Cartosat\B"
    out_root = r"D:\Rakshit\SemiCD\Cartosat\aligned"

    process_from_list(list_file, dir_A, dir_B, out_root)
