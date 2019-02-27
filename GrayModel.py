import torch
import torch.nn as nn
from torch.nn.functional import interpolate

class GrayModel(nn.Module):
    def __init__(self):
        print("Model initialized")
        super().__init__()

        # Encoder

        # ConvBlock1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),  # N x 3 x H x W --> N x 8 x H x W
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # ConvBlock2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # N x 8 x (H/2) x (W/2) --> N x 16 x (H/2) x (H/2)
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # N x 16 x (H/2) x (W/2) --> N x 16 x (H/2) x (H/2)
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # ConvBlock3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # N x 16 x (H/4) x (W/4) --> N x 32 x (H/4) x (W/4)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # N x 32 x (H/4) x (W/4) --> N x 32 x (H/4) x (W/4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # ConvBlock4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # N x 32 x (H/8) x (W/8) --> N x 64 x (H/8) x (W/8)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # N x 64 x (H/8) x (W/8) --> N x 64 x (H/8) x (W/8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Decoder
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(96, 64, 3, stride=1, padding=1),  # N x 96 x (H/4) x (W/4) --> N x 64 x (H/4) x (W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # N x 64 x (H/4) x (W/4) --> N x 32 x (H/4) x (W/4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(48, 32, 3, stride=1, padding=1),  # N x 48 x (H/2) x (W/2) --> N x 32 x (H/2) x (W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, 3, stride=1, padding=1),  # N x 32 x (H/2) x (W/2) --> N x 16 x (H/2) x (W/2)
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(24, 16, 3, stride=1, padding=1),  # N x 24 x H x W --> N x 16 x H x W
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 8, 3, stride=1, padding=1),  # N x 16 x H x W --> N x 8 x H x W
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(8, 4, 3, stride=1, padding=1),  # N x 8 x H x W --> N x 4 x H x W
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, 2, 3, stride=1, padding=1),  # N x 4 x H x W --> N x 2 x H x W
            nn.BatchNorm2d(2),
            nn.ReLU(),

            nn.Conv2d(2, 1, 3, stride=1, padding=1),  # N x 2 x H x W --> N x 1 x H x W
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, x):

        # Encode
        out1 = self.conv_block1(x)  # N x 3 x H x W --> N x 8 x H x W

        pool1_out = nn.MaxPool2d(2, 2)(out1)  # N x 8 x H x W -> N x 8 x (H/2) x (W/2)

        out2 = self.conv_block2(pool1_out)  # N x 8 x (H/2) x (W/2) --> N x 16 x (H/2) x (H/2)

        pool2_out = nn.MaxPool2d(2, 2)(out2)  # N x 16 x (H/2) x (H/2) --> N x 16 x (H/4) x (H/4)

        out3 = self.conv_block3(pool2_out)  # N x 16 x (H/4) x (W/4) --> N x 32 x (H/4) x (W/4)

        pool3_out = nn.MaxPool2d(2, 2)(out3)  # N x 32 x (H/4) x (W/4) --> N x 32 x (H/8) x (W/8)

        out4 = self.conv_block4(pool3_out)  # N x 32 x (H/8) x (W/8) --> N x 64 x (H/8) x (W/8)

        # Decode
        up1_out = interpolate(out4, scale_factor=2, mode='bilinear', align_corners=True)  # N x 64 x (H/8) x (W/8) --> N x 64 x (H/4) x (W/4)

        concat1 = torch.cat([out3, up1_out], 1)  # N x (32 + 64) x (H/4) x (W/4)

        out5 = self.conv_block5(concat1)  # N x (64 + 32) x (H/4) x (W/4) --> N x 32 x (H/4) x (W/4)

        up2_out = interpolate(out5, scale_factor=2, mode='bilinear', align_corners=True)  # N x 32 x (H/4) x (W/4) --> N x 32 x (H/2) x (W/2)

        concat2 = torch.cat([out2, up2_out], 1)  # N x (16 + 32) x (H/2) x (W/2)

        out6 = self.conv_block6(concat2)  # N x 48 x (H/4) x (W/4) --> N x 16 x (H/2) x (W/2)

        up3_out = interpolate(out6, scale_factor=2, mode='bilinear', align_corners=True)  # N x 16 x (H/2) x (W/2) --> N x 16 x H x W

        concat3 = torch.cat([out1, up3_out], 1)  # N x (8 + 16) x H x W

        out7 = self.conv_block7(concat3)  # N x 24 x H x W --> N x 8 x H x W

        out = self.conv_block8(out7)  # N x 8 x H x W --> N x 1 x H x W

        return out
