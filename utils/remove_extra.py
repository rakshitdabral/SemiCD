import os

def clean_extra_files():
    a_dir = "cartosat_data_12/A"
    b_dir = "cartosat_data_12/B"

    # Collect patch numbers (without extension)
    a_patches = {f.split("_")[1] for f in os.listdir(a_dir) if f.endswith(".tif")}
    b_patches = {f.split("_")[1] for f in os.listdir(b_dir) if f.endswith(".tif")}

    # Find common patches
    common_patches = a_patches & b_patches

    # Remove files in A not in B
    for f in os.listdir(a_dir):
        if f.endswith(".tif"):
            patch = f.split("_")[1]
            if patch not in common_patches:
                os.remove(os.path.join(a_dir, f))
                print(f"🗑 Removed extra file from A: {f}")

    # Remove files in B not in A
    for f in os.listdir(b_dir):
        if f.endswith(".tif"):
            patch = f.split("_")[1]
            if patch not in common_patches:
                os.remove(os.path.join(b_dir, f))
                print(f"🗑 Removed extra file from B: {f}")

if __name__ == "__main__":
    clean_extra_files()
