import torch
from torch import optim
from tqdm import tqdm
import os

import config
from data_loader import get_dataloaders
from sod_model import SODModel
from loss_functions import combined_loss

def train():

    #Load Data
    train_loader, val_loader, _ = get_dataloaders() # _ means ignore the test loader (we don't train with it)

    #Create model
    model = SODModel().to(config.DEVICE)

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    #every 10 epochs, multiply LR by 0.5
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    #Training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
        model.train()
        total_train_loss = 0

        #Training
        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            optimizer.zero_grad()

            preds = model(images)
            loss = combined_loss(preds, masks)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                preds = model(images)
                loss = combined_loss(preds, masks)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")

        #Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            #save best model
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
            print("Best model saved!")

        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered!")
                break

        scheduler.step()
        print(f"Current learning rate: {scheduler.get_lr()[0]:.6f}")
if __name__ == "__main__":
    train()