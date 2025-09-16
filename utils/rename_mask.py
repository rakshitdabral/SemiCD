import os

def rename_masks():
    mask_dir = "mask"

    for f in os.listdir(mask_dir):
        if f.startswith("mask_") and f.endswith(".tif"):
            # Extract patch number (the part after "mask_" and before ".tif")
            patch_num = f.split("_")[1].split(".")[0]
            new_name = f"{patch_num}.tif"

            old_path = os.path.join(mask_dir, f)
            new_path = os.path.join(mask_dir, new_name)

            os.rename(old_path, new_path)
            print(f"✅ Renamed {f} -> {new_name}")

if __name__ == "__main__":
    rename_masks()
