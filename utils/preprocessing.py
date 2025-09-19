import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os

# ----------------------------
# Feature-based homography
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

# ----------------------------
# Safety wrapper
# ----------------------------
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
        img = src.read()  # (bands, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, bands)
    img = img.astype(np.float32)

    if method == "composite":
        weights = np.array([0.1, 0.4, 0.4, 0.1], dtype=np.float32)  # B,G,R,NIR
        gray = np.tensordot(img, weights, axes=([2], [0]))
    elif method == "green":
        gray = img[:, :, 1]  # Green band
    else:
        raise ValueError("method must be 'composite' or 'green'")

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray

def read_fcc(path):
    with rasterio.open(path) as src:
        img = src.read([4, 3, 2])  # NIR, R, G
        img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

# ----------------------------
# Common alignment runner
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
# Show alignment (for visualization)
# ----------------------------
def show_alignment(ref_path, mov_path):
    for method in ["green", "composite"]:
        ref, mov, aligned, kp1, kp2, matches, mask, ok = run_alignment(ref_path, mov_path, method)

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Alignment using {method} (ok={ok})")

        plt.subplot(1, 3, 1); plt.imshow(ref); plt.title("Reference (clipped)")
        plt.subplot(1, 3, 2); plt.imshow(mov); plt.title("Moving (FCC)")
        plt.subplot(1, 3, 3); plt.imshow(aligned); plt.title("Aligned (clipped)")
        plt.show()

        if len(matches) > 0 and mask is not None:
            num_show = 30
            matches_subset = matches[:num_show]
            mask_subset = mask.ravel().tolist()[:num_show]
            match_vis = cv2.drawMatches(
                read_as_grayscale(ref_path, method), kp1,
                read_as_grayscale(mov_path, method), kp2,
                matches_subset, None,
                matchesMask=mask_subset, flags=2
            )
            plt.figure(figsize=(14, 7))
            plt.title(f"Feature Matches ({method})")
            plt.imshow(match_vis)
            plt.show()

# ----------------------------
# Save alignment (for production)
# ----------------------------
def save_alignment(ref_path, mov_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base_name_ref = os.path.splitext(os.path.basename(ref_path))[0]
    base_name_mov = os.path.splitext(os.path.basename(mov_path))[0]

    for method in ["green", "composite"]:
        ref, mov, aligned, _, _, _, _, ok = run_alignment(ref_path, mov_path, method)
        if ok:
            print(f"Saving using {method} alignment...")
            break
    else:
        # If none worked → save originals
        print("No good alignment, saving originals...")
        ref = read_fcc(ref_path)
        aligned = read_fcc(mov_path)

    ref_out = os.path.join(out_dir, f"{base_name_ref}_aligned.tif")
    mov_out = os.path.join(out_dir, f"{base_name_mov}_aligned.tif")

    with rasterio.open(ref_path) as src:
        meta = src.meta
        meta.update(count=3, dtype=rasterio.uint8)

        with rasterio.open(ref_out, "w", **meta) as dst:
            dst.write(np.transpose(ref, (2, 0, 1)).astype(np.uint8))

        with rasterio.open(mov_out, "w", **meta) as dst:
            dst.write(np.transpose(aligned, (2, 0, 1)).astype(np.uint8))

    print(f"Saved:\n  {ref_out}\n  {mov_out}")


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    ref = r"D:\hitesh\encroachment montitoring\patches1\patch_00001_T1.tif"
    mov = r"D:\hitesh\encroachment montitoring\patches1\patch_00001_T2.tif"

    show_alignment(ref, mov)   # interactive
    #save_alignment(ref, mov, r"D:\hitesh\encroachment montitoring\aligned")