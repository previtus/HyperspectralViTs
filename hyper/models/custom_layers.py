# Model proposals and adaptations by Vit Ruzicka (in 2024-2025 during PhD)
# Relevant paper: https://doi.org/10.1109/JSTARS.2025.3557527
# - Custom layers for Upscaling block in the proposed models

from torch import nn
import torch.nn.functional as F

class UpscaleBlock(nn.Module):
    # Upscale block that will increase the spatial resolution at the cost of the number of channels (containing latent feature).
    # Effectively should be able to do a learned 2x upscale. (Note: out_channels might be good as half of in_channels)
    def __init__(
        self,
        in_channels=256,
        out_channels=128,
        num_layers=2,
        use_batchnorm=True,
    ):
        super().__init__()

        self.num_layers = num_layers
        if self.num_layers > 3:
            print("Only 1 or 2 supported as UpscaleBlock.num_layers...")

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.relu1 = nn.ReLU(inplace=True)

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Identity()

        # the rest of the layers should have the smaller dim
        in_channels = out_channels

        if self.num_layers == 2:
            self.conv2 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            )
            self.relu2 = nn.ReLU(inplace=True)

            if use_batchnorm:
                self.bn2 = nn.BatchNorm2d(out_channels)
            else:
                self.bn2 = nn.Identity()


    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        if self.num_layers == 2:
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.bn2(x)

        return x

