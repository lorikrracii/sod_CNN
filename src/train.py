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
    train_loader, val_loader, _ = get_dataloaders()

    #Create model
    model = SODModel().to(config.DEVICE)

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

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