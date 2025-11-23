import os
import shutil

SOURCE = "../data/raw/all/"
DEST_IMAGES = "../data/raw/images/"
DEST_MASKS = "../data/raw/masks/"

os.makedirs(DEST_IMAGES, exist_ok=True)
os.makedirs(DEST_MASKS, exist_ok=True)

files = os.listdir(SOURCE)

for f in files:
    if f.lower().endswith(".jpg"):
        shutil.move(os.path.join(SOURCE, f), os.path.join(DEST_IMAGES, f))
    elif f.lower().endswith(".png"):
        shutil.move(os.path.join(SOURCE, f), os.path.join(DEST_MASKS, f))

print("MSRA10K dataset sorted successfully!")
