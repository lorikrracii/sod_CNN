import torch
from loss_functions import combined_loss

pred = torch.rand(2, 1, 128, 128)
target = torch.rand(2, 1, 128, 128)

loss = combined_loss(pred, target)
print("Loss:", loss.item())
