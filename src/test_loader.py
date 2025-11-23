from data_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

images, masks = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Mask batch shape:", masks.shape)
print("Image value range:", images.min().item(), "to", images.max().item())
print("Mask value range:", masks.min().item(), "to", masks.max().item())
