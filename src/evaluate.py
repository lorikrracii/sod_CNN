import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

import config
from data_loader import SODDataset
from sod_model import SODModel

#metric functions
def iou_score(pred, mask, eps=1e-6):
    pred = pred.view(-1)
    mask = mask.view(-1)

    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection

    return (intersection + eps) / (union + eps)

def precision_score(pred, mask, eps=1e-6):
    pred = pred.view(-1)
    mask = mask.view(-1)

    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()

    return (tp + eps) / (tp + fp + eps)

def recall_score(pred, mask, eps=1e-6):
    pred = pred.view(-1)
    mask = mask.view(-1)

    tp = (pred * mask).sum()
    fn = ((1 - pred) * mask).sum()

    return (tp + eps) / (tp + fn + eps)


def f1_score(pred, mask, eps=1e-6):
    p = precision_score(pred, mask, eps)
    r = recall_score(pred, mask, eps)
    return 2 * (p * r) / (p + r + eps)


def mae_score(pred, mask):
    return torch.abs(pred - mask).mean()

#EVALUATION
def evaluate():

    print("\nLoading model...")
    model = SODModel().to(config.DEVICE)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    print("Model loaded successfully.")

    print("\nLoading test dataset...")
    test_dataset = SODDataset(config.TEST_IMG_DIR, config.TEST_MASK_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    iou_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    mae_list = []

    print("\nEvaluating on test set..")

    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            preds = model(images)
            preds = (preds > 0.5).float()

            #compute metrics
            iou_list.append(iou_score(preds, masks).item())
            prec_list.append(precision_score(preds, masks).item())
            rec_list.append(recall_score(preds, masks).item())
            f1_list.append(f1_score(preds, masks).item())
            mae_list.append(mae_score(preds, masks).item())

    #averages
    iou = np.mean(iou_list)
    prec = np.mean(prec_list)
    rec = np.mean(rec_list)
    f1 = np.mean(f1_list)
    mae = np.mean(mae_list)

    print("\n------------------ MODEL PERFORMANCE ------------------")
    print(f"IoU:        {iou:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"MAE:        {mae:.4f}")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    evaluate()