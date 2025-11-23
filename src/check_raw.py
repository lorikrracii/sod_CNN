import os
import config

print("RAW IMAGES:", len(os.listdir(config.RAW_IMG_DIR)))
print("RAW MASKS :", len(os.listdir(config.RAW_MASK_DIR)))
