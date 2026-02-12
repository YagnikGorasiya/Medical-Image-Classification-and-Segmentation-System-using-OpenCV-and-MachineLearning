import os
import shutil
import random

source_dir = "datasets/brain_mri_raw"
target_dir = "datasets/brain_mri"

split_ratio = (0.7, 0.15, 0.15)

classes = os.listdir(source_dir)

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split in splits:
        class_name = "TUMOR" if cls == "yes" else "NO_TUMOR"
        os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)

        for img in splits[split]:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(target_dir, split, class_name, img)
            )

print("Brain dataset split complete!")

