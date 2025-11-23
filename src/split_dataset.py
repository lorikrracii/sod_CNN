import os
import shutil
import numpy as np
import config

def main():
    # sigurohemi qe destination exists
    for path in [
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, config.VAL_MASK_DIR,
        config.TEST_IMG_DIR, config.TEST_MASK_DIR,
    ]:
        os.makedirs(path, exist_ok=True)

    #i listojme images dhe masks
    images = sorted(os.listdir(config.RAW_IMG_DIR))
    masks = sorted(os.listdir(config.RAW_MASK_DIR))

    assert len(images) == len(masks), "Number of images and masks must match"

    n = len(images)
    indices = np.arange(n)
    np.random.seed(config.SEED)
    np.random.shuffle(indices)

    # compute split points : 70% train, 15% validation, 15% test

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def copy_subset(idx_list, img_dest, mask_dest):
        for idx in idx_list:
            img_name = images[idx]
            mask_name = masks[idx]

            #same base name
            src_img = os.path.join(config.RAW_IMG_DIR, img_name)
            src_mask = os.path.join(config.RAW_MASK_DIR, mask_name)

            dst_img = os.path.join(img_dest, img_name)
            dst_mask = os.path.join(mask_dest, mask_name)

            shutil.copy(src_img, dst_img)
            shutil.copy(src_mask, dst_mask)

    # copy files to split folders
    copy_subset(train_idx, config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR)
    copy_subset(val_idx, config.VAL_IMG_DIR, config.VAL_MASK_DIR)
    copy_subset(test_idx, config.TEST_IMG_DIR, config.TEST_MASK_DIR)

    print(f"Total samples: {n}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print("Dataset successfully split into train , val , test !! ")
    print("Example raw image:", images[0])
    print("Example raw mask:", masks[0])


if __name__ == "__main__":
    main()