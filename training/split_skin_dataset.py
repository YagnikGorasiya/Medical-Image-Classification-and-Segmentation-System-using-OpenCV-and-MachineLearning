import os, shutil, random

source_dir = "datasets/skin_lesion_raw"
target_dir = "datasets/skin_lesion"
split_ratio = (0.7, 0.15, 0.15)

# Detect class folders automatically
classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

# Normalize mapping
def normalize_class(name: str) -> str:
    n = name.strip().lower()
    if "ben" in n:
        return "BENIGN"
    if "mal" in n:
        return "MALIGNANT"
    return name.upper()

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    out_cls = normalize_class(cls)

    for split in splits:
        out_dir = os.path.join(target_dir, split, out_cls)
        os.makedirs(out_dir, exist_ok=True)
        for img in splits[split]:
            shutil.copy(os.path.join(cls_path, img), os.path.join(out_dir, img))

print("Skin dataset split complete!")
print("Classes found:", classes)
