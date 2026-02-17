import os
import random
import shutil

SOURCE_DIR = "data/real_vs_fake/real-vs-fake"
TARGET_DIR = "data/small_dataset"

LIMIT_PER_CLASS = 5000  # change if needed

def copy_limited_images(split):
    for label in ["real", "fake"]:
        source_path = os.path.join(SOURCE_DIR, split, label)
        target_path = os.path.join(TARGET_DIR, split, label)

        os.makedirs(target_path, exist_ok=True)

        images = os.listdir(source_path)
        selected = random.sample(images, min(LIMIT_PER_CLASS, len(images)))

        for img in selected:
            shutil.copy(
                os.path.join(source_path, img),
                os.path.join(target_path, img)
            )

for split in ["train", "valid", "test"]:
    copy_limited_images(split)

print("Small dataset created successfully!")
