import os
import random
from collections import defaultdict

def create_semisupervised_lists(
    root="cartosat",
    a_dir="A",
    b_dir="B",
    label_dir="label",
    list_dir="list",
    seed=42,
    train_sup_ratio=0.6,
    val_sup_ratio=0.2,
    test_sup_ratio=0.2,
):
    """
    Build splits:
      - Supervised = stems present in A, B, and label
      - Unsupervised = stems present in A and B but NOT in label

    Writes list files with filenames (basename + extension).
    """

    random.seed(seed)

    root = os.path.abspath(root)
    A_path     = os.path.join(root, a_dir)
    B_path     = os.path.join(root, b_dir)
    LABEL_path = os.path.join(root, label_dir)
    LIST_path  = os.path.join(root, list_dir)
    os.makedirs(LIST_path, exist_ok=True)

    for p in [A_path, B_path, LABEL_path]:
        if not os.path.isdir(p):
            print(f"Error: missing directory: {p}")
            return

    # --- Helpers ---
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}

    def scan_dir(d):
        """Return dict: stem -> list of full filenames (with ext) found in d."""
        found = defaultdict(list)
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if not os.path.isfile(fp):
                continue
            name, ext = os.path.splitext(f)
            if ext.lower() in exts:
                found[name].append(f)  # store filename with extension
        return found

    def pick_filename(stem, bucket):
        """Prefer .png if multiple versions exist; else pick first alphabetically."""
        candidates = sorted(bucket[stem])
        pngs = [c for c in candidates if c.lower().endswith(".png")]
        return pngs[0] if pngs else candidates[0]

    # --- Index all three folders by stem ---
    A_idx = scan_dir(A_path)
    B_idx = scan_dir(B_path)
    L_idx = scan_dir(LABEL_path)

    stems_A   = set(A_idx.keys())
    stems_B   = set(B_idx.keys())
    stems_L   = set(L_idx.keys())

    # Pairs that exist in both A and B
    stems_pairs = stems_A & stems_B
    # Supervised = A ∩ B ∩ label
    stems_sup   = stems_pairs & stems_L
    # Unsupervised = (A ∩ B) \ label
    stems_unsup = stems_pairs - stems_L

    # Materialize filenames (choose an extension per stem consistently per folder)
    sup_files = []
    for s in sorted(stems_sup):
        a_file = pick_filename(s, A_idx)
        b_file = pick_filename(s, B_idx)
        l_file = pick_filename(s, L_idx)
        # We only write the shared "key" (usually the A filename). Most loaders use the same name in A/B/label.
        # If your repo expects EXACT same name in all three dirs, ensure your masks follow that.
        # Here we standardize on the A filename for the lists.
        # (You can switch to l_file if your repo keys from label names.)
        sup_files.append(a_file)

    unsup_files = []
    for s in sorted(stems_unsup):
        a_file = pick_filename(s, A_idx)
        b_file = pick_filename(s, B_idx)
        unsup_files.append(a_file)

    # Deduplicate just in case
    sup_files   = sorted(list(dict.fromkeys(sup_files)))
    unsup_files = sorted(list(dict.fromkeys(unsup_files)))

    # --- Shuffle for splits ---
    random.shuffle(sup_files)
    random.shuffle(unsup_files)

    # --- Supervised train/val/test splits ---
    n_sup = len(sup_files)
    n_train_sup = int(n_sup * train_sup_ratio)
    n_val_sup   = int(n_sup * val_sup_ratio)
    # remainder to test
    n_test_sup  = n_sup - n_train_sup - n_val_sup

    train_supervised = sup_files[:n_train_sup]
    val_supervised   = sup_files[n_train_sup:n_train_sup+n_val_sup]
    test_supervised  = sup_files[n_train_sup+n_val_sup:]

    # --- Unsupervised: typically train-only ---
    train_unsupervised = unsup_files

    # --- Writer ---
    def write_list(name, arr):
        path = os.path.join(LIST_path, name)
        with open(path, "w", encoding="utf-8") as f:
            for x in arr:
                f.write(f"{x}\n")
        print(f"Created: {path} ({len(arr)} items)")

    # Save files
    write_list("train_supervised.txt", train_supervised)
    write_list("val_supervised.txt",   val_supervised)
    write_list("test_supervised.txt",  test_supervised)
    write_list("train_unsupervised.txt", train_unsupervised)

    # Optional: full listings for debugging/inspection
    write_list("all_supervised.txt", sup_files)
    write_list("all_unsupervised.txt", unsup_files)

    # --- Summary ---
    print("\nSummary")
    print("-------")
    print(f"Root: {root}")
    print(f"Pairs (A∩B):                 {len(stems_pairs)}")
    print(f"Supervised (A∩B∩label):      {len(sup_files)}")
    print(f"Unsupervised (A∩B \\ label):  {len(unsup_files)}")
    print("\nSupervised split:")
    print(f"  train: {len(train_supervised)}")
    print(f"  val:   {len(val_supervised)}")
    print(f"  test:  {len(test_supervised)}")
    print("\nUnsupervised split:")
    print(f"  train: {len(train_unsupervised)}")

    # Show a few examples
    def peek(tag, arr):
        print(f"  {tag} example: {arr[0]}" if arr else f"  {tag}: (empty)")

    print("\nExamples")
    peek("train_supervised", train_supervised)
    peek("val_supervised",   val_supervised)
    peek("test_supervised",  test_supervised)
    peek("train_unsupervised", train_unsupervised)

if __name__ == "__main__":
    create_semisupervised_lists()
