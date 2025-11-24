import torch
import torch.nn as nn
from cv2.gapi import kernel

import config

class ConvBlock(nn.Module):

    #basic block : conv2d + relu
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SODModel(nn.Module):
    #encoder - decoder CNN
    def __init__(self):
        super().__init__()

        enc_channels = config.ENCODER_CHANNELS
        dec_channels = config.DECODER_CHANNELS

        #encoder
        self.enc1 = ConvBlock(3, enc_channels[0])
        self.enc2 = ConvBlock(enc_channels[0], enc_channels[1])
        self.enc3 = ConvBlock(enc_channels[1], enc_channels[2])

        # 1 maxpool layer reused 3 times
        self.pool = nn.MaxPool2d(   kernel_size = 2, stride=2)

        # decoder
        # ConvTranspose2d doubles H and W when kernel_size = 2, stride = 2
        # decoder (fixed)
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 32x32 -> 64x64
        self.dec3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)  # 64x64 -> 128x128

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.pool(x)

        x = self.enc2(x)
        x = self.pool(x)

        x = self.enc3(x)
        x = self.pool(x)

        #Decoder
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        x = self.dec3(x)

        x = self.sigmoid(x)

        return x