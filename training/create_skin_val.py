import os, shutil, random

train_dir = "datasets/skin_lesion/train"
val_dir = "datasets/skin_lesion/val"

split_ratio = 0.15

classes = os.listdir(train_dir)

for cls in classes:
    cls_path = os.path.join(train_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    n_val = int(len(images) * split_ratio)

    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in images[:n_val]:
        shutil.move(
            os.path.join(cls_path, img),
            os.path.join(val_dir, cls, img)
        )

print("Validation split created successfully!")
