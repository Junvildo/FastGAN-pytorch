import torch
import torch.nn as nn
import time

# Parameters
nz = 100
channels = 64
batch_size = 8

# Create noise input
noise = torch.FloatTensor(batch_size, nz).normal_(0, 1)
noise = noise.view(noise.shape[0], -1, 1, 1)

# Define ConvTranspose2d layer
conv_transpose = nn.ConvTranspose2d(nz, channels * 2, 4, 1, 0, bias=False)

# Apply the layer
n = 10000
t0 = time.time()
for i in range(n): output = conv_transpose(noise)
t1 = time.time()

total_n = t1-t0
print(total_n/n)
# print(output.shape)  # Expected shape: (8, 128, 4, 4)

import torch
import torch.nn as nn

# Parameters
nz = 100
channels = 64
batch_size = 8
upscale_factor = 4

# Create noise input
noise = torch.FloatTensor(batch_size, nz).normal_(0, 1)
noise = noise.view(noise.shape[0], -1, 1, 1)  # Shape: (8, 100, 1, 1)

# Define Conv2d and PixelShuffle layers
conv = nn.Conv2d(nz, 128 * (upscale_factor ** 2), kernel_size=1, stride=1, padding=0, bias=False)
pixel_shuffle = nn.PixelShuffle(upscale_factor)

# Define a simple model combining the conv and pixel shuffle layers
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = conv
        self.pixel_shuffle = pixel_shuffle

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

# Instantiate and apply the model
model = SimpleModel()
n = 10000
t0 = time.time()
for i in range(n): output = model(noise)
t1 = time.time()

total_n = t1-t0
print(total_n/n)
# print(output.shape)  # Expected shape: (8, 128, 4, 4)


import torch
import torch.nn as nn

# Parameters
batch_size = 8
in_channels = 64
H, W = 224, 224

# Example input tensor
input_tensor = torch.randn(batch_size, in_channels, H, W)

# Original Upsample layer
upsample = nn.Upsample(scale_factor=2, mode='nearest')
n = 100
t0 = time.time()
for i in range(n): upsample_output = upsample(input_tensor)
t1 = time.time()

total_n = t1-t0
print(total_n/n)
# Upsample output
print(upsample_output.shape)  # Expected shape: (8, 64, 32, 32)


# Define Conv2d and PixelShuffle layers
conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

# Define a simple model combining the conv and pixel shuffle layers
class UpsampleWithPixelShuffle(nn.Module):
    def __init__(self):
        super(UpsampleWithPixelShuffle, self).__init__()
        self.conv = conv
        self.pixel_shuffle = pixel_shuffle

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

# Instantiate and apply the model
model = UpsampleWithPixelShuffle()
n = 100
t0 = time.time()
for i in range(n):  output = model(input_tensor)
t1 = time.time()

total_n = t1-t0
print(total_n/n)
print(output.shape)  # Expected shape: (8, 64, 32, 32)
