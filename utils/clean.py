# Paths to your files
train_file = r"Cartosat\list\100_train_supervised.txt"
val_file   = r"Cartosat\list\val.txt"

# Read both files
with open(train_file, "r") as f:
    train_images = set(line.strip() for line in f if line.strip())

with open(val_file, "r") as f:
    val_images = set(line.strip() for line in f if line.strip())

# Remove overlaps from training
cleaned_train_images = train_images - val_images

# Save back the cleaned train file
with open(train_file, "w") as f:
    for img in sorted(cleaned_train_images):
        f.write(img + "\n")

print(f"Removed {len(train_images - cleaned_train_images)} overlapping images.")
print(f"Final train count: {len(cleaned_train_images)}")
print(f"Validation count (unchanged): {len(val_images)}")
