import torch
from  sod_model import SODModel
import config

if __name__ == '__main__':
    model = SODModel()
    model.eval()

    x = torch.randn(4,3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape :", y.shape)
    print("Output range:", y.min().item(), "to", y.max().item())