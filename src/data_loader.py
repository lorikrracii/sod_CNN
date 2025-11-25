import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config

class SODDataset(Dataset):

    #loads images and masks, pairs and preprocesses them

    def __init__(self, img_dir, mask_dir, augment = False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert(len(self.images) == len(self.masks))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        # load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        #AUGMENTATION
        if self.augment:
            #1 random horizontal flip
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

            #2 random brightness & contrast
            if np.random.rand() > 0.5:
                alpha = 0.8 + np.random.rand() * 0.4 #contrast factor between 0.8-1.2
                beta = np.random.randint(-20,20)    #brightness shift
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            #3 small gaussian blur
            if np.random.rand() > 0.7:
                image = cv2.GaussianBlur(image, (3, 3), 0)

        #resize after augmentation
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        mask = cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        # normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)

        # mask: (H,W) -> (1,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask



def get_dataloaders():

    train_dataset = SODDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,augment=True )
    val_dataset = SODDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR,augment=False)
    test_dataset = SODDataset(config.TEST_IMG_DIR, config.TEST_MASK_DIR,augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader