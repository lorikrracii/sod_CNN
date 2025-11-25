import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from sod_model import SODModel
from data_loader import SODDataset

# Ensure results/visuals exists
VIS_DIR = os.path.join(config.RESULT_DIR, "visuals")
os.makedirs(VIS_DIR, exist_ok=True)


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    Create transparent overlay of predicted mask on image. 2d array with values 0-1
    """
    mask_bool = mask > 0.5

    overlay = image.copy()

    # Apply color per-channel (robust for NumPy broadcasting)
    overlay[mask_bool, 0] = color[0]
    overlay[mask_bool, 1] = color[1]
    overlay[mask_bool, 2] = color[2]

    # Blend with original image
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)



def visualize_predictions(num_samples=5):
    print("\nLoading model...")
    model = SODModel().to(config.DEVICE)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    print("Model loaded successfully")

    test_dataset = SODDataset(config.TEST_IMG_DIR, config.TEST_MASK_DIR)
    test_len = len(test_dataset)

    print(f"\nGenerating visualizations for {num_samples} random samples...\n")

    for i in range(num_samples):
        idx = np.random.randint(0, test_len)
        image, mask = test_dataset[idx]

        # Prepare for prediction
        image_tensor = image.unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            pred = model(image_tensor)
            pred = (pred > 0.5).float().cpu().numpy()[0, 0]  # stays 0–1

        # Convert tensors for visualization
        img_np = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        mask_np = (mask.numpy()[0] * 255).astype(np.uint8)
        pred_vis = (pred * 255).astype(np.uint8)

        # Create overlay **using pred 0–1 mask**
        overlay = overlay_mask(img_np, pred)

        # Create visual grid
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(pred_vis, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        save_path = os.path.join(VIS_DIR, f"sample_{i+1}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

    print("\nVisualization complete! Check results/visuals/ folder.\n")


if __name__ == "__main__":
    visualize_predictions(num_samples=5)
