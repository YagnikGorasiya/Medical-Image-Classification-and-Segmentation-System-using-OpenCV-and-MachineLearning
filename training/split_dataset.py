import os
import shutil
import random

def split_dataset(source_dir, target_dir, split_ratio=(0.7, 0.15, 0.15)):
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
            os.makedirs(os.path.join(target_dir, split, cls.upper()), exist_ok=True)

            for img in splits[split]:
                shutil.copy(
                    os.path.join(cls_path, img),
                    os.path.join(target_dir, split, cls.upper(), img)
                )

    print("Dataset split complete!")

# Example usage
split_dataset(
    source_dir="datasets/brain_mri_raw",
    target_dir="datasets/brain_mri"
)
